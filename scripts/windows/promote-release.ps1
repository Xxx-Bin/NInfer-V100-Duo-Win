<#
.SYNOPSIS
    Promotes a prerelease test build into a real GitHub release, in place.

.DESCRIPTION
    The release flow publishes `test-<tag>` as a prerelease, acceptance is run on the V100 host, and
    this script turns the accepted build into the real release without rebuilding or re-uploading the
    archive:

      1. rename the tag           test-<tag> -> <tag>
      2. clear `prerelease` and drop the `[TEST]` title prefix
      3. replace the TEST BUILD banner in the body with the acceptance note
      4. rename the assets        ninfer-test-<tag>-... -> ninfer-<tag>-...
      5. rewrite the .sha256 sidecar so it names the renamed archive
      6. delete the leftover test tag ref (renaming a tag through the API leaves the old ref behind)

    Everything goes through the Releases API: GitHub refuses a `git push` that creates a ref whose
    commit touches `.github/workflows/*`.

.PARAMETER Tag
    The real tag to promote to, e.g. `v0.1.1-win.6`. The prerelease is looked up as `test-<Tag>`.

.PARAMETER AcceptanceNote
    One or two Markdown lines recording what was verified for this build. They replace the
    `> **TEST BUILD**` banner in the published body.

.PARAMETER DryRun
    Print the planned changes and stop before the first write.

.EXAMPLE
    $env:GH_TOKEN = (Get-Content _scratch\gh_token.txt -Raw).Trim()
    scripts\windows\promote-release.ps1 -Tag v0.1.1-win.6 -AcceptanceNote 'Accepted on 2x V100: media smoke 10/10, TP2 text check, archive extract.' -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Tag,
    [string] $Repository = 'Xxx-Bin/NInfer-V100-Duo-Win',
    [string] $Token = $env:GH_TOKEN,
    [string] $AcceptanceNote,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $Token) {
    throw 'no token: pass -Token or set GH_TOKEN'
}
if ($Tag -match '^test-') {
    throw "-Tag takes the real tag (vX.Y.Z-win.N), not the test tag: got '$Tag'"
}

$testTag = "test-$Tag"
$api = "https://api.github.com/repos/$Repository"
$headers = @{
    Authorization          = "Bearer $Token"
    Accept                 = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
}

function Get-ErrorDetail {
    param([Parameter(Mandatory)] $ErrorRecord)
    # Windows PowerShell surfaces the body through the WebResponse stream, PowerShell 7 through
    # HttpResponseMessage.Content; a bare status code hides the API's own message.
    $details = $ErrorRecord.ErrorDetails
    if ($null -ne $details -and $details.Message) { return $details.Message }
    $response = $ErrorRecord.Exception.Response
    if ($null -eq $response) { return $ErrorRecord.Exception.Message }
    try {
        if ($null -ne $response.Content) {
            $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if ($text) { return $text }
        }
    } catch {
        Write-Verbose "no Content on the error response: $($_.Exception.Message)"
    }
    try {
        $stream = $response.GetResponseStream()
        if ($null -ne $stream) {
            $text = (New-Object System.IO.StreamReader($stream)).ReadToEnd()
            if ($text) { return $text }
        }
    } catch {
        Write-Verbose "no response stream: $($_.Exception.Message)"
    }
    return $ErrorRecord.Exception.Message
}

function Invoke-Api {
    param(
        [Parameter(Mandatory)][string] $Method,
        [Parameter(Mandatory)][string] $Uri,
        [object] $Body,
        [byte[]] $Bytes,
        [string] $ContentType = 'application/json'
    )
    $attempt = 0
    while ($true) {
        $attempt++
        try {
            $arguments = @{
                Method      = $Method
                Uri         = $Uri
                Headers     = $headers
                TimeoutSec  = 60
                ErrorAction = 'Stop'
            }
            if ($Bytes) {
                $arguments.Body        = $Bytes
                $arguments.ContentType = 'application/octet-stream'
            } elseif ($null -ne $Body) {
                $arguments.Body        = ($Body | ConvertTo-Json -Depth 5 -Compress)
                $arguments.ContentType = $ContentType
            }
            $response = Invoke-RestMethod @arguments
            return $response
        } catch {
            $status = $null
            try { $status = [int]$_.Exception.Response.StatusCode } catch { $status = $null }
            $detail = Get-ErrorDetail -ErrorRecord $_
            if ($status -in 401, 403, 404, 422) {
                throw "GitHub $Method $Uri -> $status : $detail"
            }
            if ($attempt -ge 8) {
                throw "GitHub $Method $Uri failed after $attempt attempts: $detail"
            }
            Write-Verbose "retry $attempt ($status): $detail"
            Start-Sleep -Seconds 8
        }
    }
}

Write-Host "looking up prerelease $testTag"
$release = Invoke-Api -Method GET -Uri "$api/releases/tags/$testTag"
if (-not $release.prerelease) {
    throw "$testTag is not a prerelease; refusing to touch it"
}
$releaseId = $release.id
$zipAsset = $release.assets | Where-Object { $_.name -like '*.zip' } | Select-Object -First 1
$sidecarAsset = $release.assets | Where-Object { $_.name -like '*.zip.sha256' } | Select-Object -First 1
if (-not $zipAsset -or -not $sidecarAsset) {
    throw "$testTag does not carry both an archive and a .sha256 sidecar"
}

$newZipName = $zipAsset.name -replace '^ninfer-test-', 'ninfer-'
if ($newZipName -notmatch [regex]::Escape($Tag)) {
    throw "archive name '$($zipAsset.name)' does not carry the test tag '$Tag'"
}
$newSidecarName = "$newZipName.sha256"
$newTitle = if ($release.name) { $release.name -replace '^\[TEST\]\s*', '' } else { "NInfer $Tag" }

$rawBody = if ($null -ne $release.body) { $release.body } else { '' }
$body = $rawBody -split "`r?`n"
if ($AcceptanceNote) {
    $body = $body | ForEach-Object {
        if ($_ -match '^> \*\*TEST BUILD\*\*') { $AcceptanceNote } else { $_ }
    }
} else {
    $body = $body | Where-Object { $_ -notmatch '^> \*\*TEST BUILD\*\*' }
}
$newBody = ($body -join "`n").TrimEnd()
# The body names the assets, so the renames have to reach the text too. The sidecar name contains
# the archive name, so it is replaced first.
$newBody = $newBody.Replace($sidecarAsset.name, $newSidecarName).Replace($zipAsset.name, $newZipName)

Write-Host "promote $testTag -> $Tag"
Write-Host "  release id   $releaseId"
Write-Host "  title        $newTitle"
Write-Host "  archive      $($zipAsset.name) -> $newZipName"
Write-Host "  sidecar      $($sidecarAsset.name) -> $newSidecarName"
Write-Host "  body banner  $(if ($AcceptanceNote) { 'replaced with the acceptance note' } else { 'dropped' })"
if ($DryRun) {
    Write-Host "--- body after promotion ---"
    Write-Host $newBody
    Write-Host "dry run: nothing was written"
    return
}

# 1-3: tag rename, prerelease flag, title and body.
$null = Invoke-Api -Method PATCH -Uri "$api/releases/$releaseId" -Body @{
    tag_name   = $Tag
    name       = $newTitle
    prerelease = $false
    body       = $newBody
    make_latest = 'true'
}
Write-Host "tag renamed to $Tag, prerelease cleared"

# 4: asset names. The archive keeps its bytes and its digest.
$null = Invoke-Api -Method PATCH -Uri "$api/releases/assets/$($zipAsset.id)" -Body @{ name = $newZipName }
$null = Invoke-Api -Method PATCH -Uri "$api/releases/assets/$($sidecarAsset.id)" -Body @{ name = $newSidecarName }
Write-Host "assets renamed"

# 5: the sidecar is a text file, so its content has to be replaced rather than renamed.
$digest = $zipAsset.digest
if (-not $digest) {
    throw "asset $($zipAsset.name) has no digest to carry over"
}
$sha = $digest -replace '^sha256:', ''
$null = Invoke-Api -Method DELETE -Uri "$api/releases/assets/$($sidecarAsset.id)"
$payload = [System.Text.Encoding]::ASCII.GetBytes("$sha  $newZipName")
$null = Invoke-Api -Method POST `
    -Uri "https://uploads.github.com/repos/$Repository/releases/$releaseId/assets?name=$newSidecarName" `
    -Bytes $payload
Write-Host "sidecar rewritten ($sha)"

# 6: the API rename created the new ref and left the test ref in place.
try {
    $null = Invoke-Api -Method DELETE -Uri "$api/git/refs/tags/$testTag"
    Write-Host "test tag $testTag deleted"
} catch {
    Write-Warning "could not delete $testTag (already gone?): $($_.Exception.Message)"
}

$final = Invoke-Api -Method GET -Uri "$api/releases/$releaseId"
Write-Host "done: $($final.tag_name) prerelease=$($final.prerelease)"
$final.assets | ForEach-Object { Write-Host "  asset $($_.name) $($_.size) B $($_.digest)" }
