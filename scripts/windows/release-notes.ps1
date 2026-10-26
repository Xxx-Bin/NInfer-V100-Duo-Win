<#
.SYNOPSIS
    Writes the GitHub release body for a Windows x64 media build.

.DESCRIPTION
    The body is produced as a file instead of a multiline `body: |` workflow input. A block scalar
    carrying a multi-row Markdown table went through YAML and env-file interpolation, and the table
    arrived as one table per row, so the text now never touches either.

    The archive name, its SHA256, the staged file list and the CI identity all come from the caller;
    the description of the build is fixed by the pipeline itself.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ZipName,
    [Parameter(Mandatory)][string] $Sha256,
    [Parameter(Mandatory)][string] $DistDirectory,
    [Parameter(Mandatory)][string] $OutFile,
    [switch] $Test
)

$ErrorActionPreference = 'Stop'

$rows = Get-ChildItem $DistDirectory -File | Sort-Object Name | ForEach-Object {
    "| ``$($_.Name)`` | $([math]::Round($_.Length / 1MB, 1)) MiB |"
}
$banner = if ($Test) {
    '> **TEST BUILD** - prefixed to distinguish it from a real release. The binaries are produced by the real pipeline; only the tag carries the test prefix.'
} else {
    ''
}
$runUrl = "$env:GITHUB_SERVER_URL/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"

# Single-quoted here-string: backticks stay literal, which is what the Markdown code spans need.
$notes = @'
Windows x64 build for two Tesla V100 (sm_70) with the media path enabled:
`NINFER_STUB_MEDIA=OFF` - FFmpeg 8.1 image/video decode plus libcurl 8.11.1 URL
acquisition, so `--vision` and `image_url`/`video_url` prompts work.

| | |
|---|---|
| Toolchain | MSVC 19.x (Visual Studio 2022), CUDA 12.8.61 (portable redist), CMake + Ninja |
| Runner | `windows-2022`, `NINFER_BUILD_JOBS=4` |
| Commit | `__COMMIT__` |
| Run | __RUN_URL__

## Performance

Measured on the target hardware: 2 x Tesla V100-SXM2-16 GB, tensor-parallel `--tp 2`,
`--kv-dtype int8`, prompt 60000 / generation 512, two repetitions per configuration.

| Metric | Baseline `34412f4f` | This build | Delta |
|---|---:|---:|---:|
| Prefill | 945.799 tok/s | **1187.077 tok/s** | **+25.5%** |
| Decode | 129.727 tok/s | **135.786 tok/s** | **+4.7%** |

Both improvements keep the A-class oracle bit-identical to the baseline
(`sha1 06287348665d3648`: accepted/drafted/rounds 406/421/106, acceptance length 4.8301887 on
both repetitions), so the speedup is not bought with a numerical change:

- **Decode +4.5%** - two-epoch packet mailbox for collectives up to 80 KiB, replacing the
  one-shot path for the small messages decode actually sends.
- **Prefill +25.3%** - the wide-T down projection routes its plain row-parallel rank through
  `nvfp4_cutlass_sm70` instead of the wide-T mma kernel; phase attribution showed the two ranks
  differed 4x (1.9 ms vs 8.1 ms) at the same shape.

__BANNER__

Unzip the archive into one directory and run the executables from there: the media
build loads FFmpeg and libcurl from the executable directory, so the DLLs must stay
beside `ninfer.exe` and `ninfer-serve.exe`.

| Archive | SHA256 |
|---|---|
| `__ZIP__` | `__SHA__` |

`__ZIP__.sha256` is the sha256sum-style sidecar for the same archive.

| Contents | Size |
|---|---|
__CONTENTS__
'@

$notes = $notes.
    Replace('__COMMIT__', $env:GITHUB_SHA).
    Replace('__RUN_URL__', $runUrl).
    Replace('__BANNER__', $banner).
    Replace('__ZIP__', $ZipName).
    Replace('__SHA__', $Sha256).
    Replace('__CONTENTS__', ($rows -join "`n"))

# LF only: the release body is Markdown, and CRLF would leave stray carriage returns in it.
$text = (($notes -split "`r?`n") -join "`n").TrimEnd() + "`n"
[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) $OutFile),
    $text,
    (New-Object System.Text.UTF8Encoding($false))
)
Write-Output "release notes written to $OutFile ($($rows.Count) staged files)"
