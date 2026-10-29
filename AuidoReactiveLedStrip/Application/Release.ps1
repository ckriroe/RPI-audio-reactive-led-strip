param(
    [string]$increment = ""  # patch | minor | major | custom, or empty for interactive prompt
)

$ErrorActionPreference = "Stop"

# Ensure script runs from its directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (![string]::IsNullOrEmpty($scriptDir)) {
    Set-Location $scriptDir
}

# ===== CONFIG =====
$publishDir = "bin\Release\net8.0\publish"
$publishProfile = "Properties\PublishProfiles\PublishRaspiFiles.pubxml"
$versionFile = "version.txt"
$csprojFile = "Application.csproj"

# ===== VERSIONING =====
if (!(Test-Path $versionFile)) {
    "0.0.0" | Set-Content $versionFile
}

$version = (Get-Content $versionFile).Trim()
$parts = $version.Split('.')

[int]$major = $parts[0]
[int]$minor = $parts[1]
[int]$patch = $parts[2]

$oldVersion = "$major.$minor.$patch"
$patchVer = "$major.$minor.$($patch + 1)"
$minorVer = "$major.$($minor + 1).0"
$majorVer = "$($major + 1).0.0"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       RPISC Release Publisher          " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Current version: v$oldVersion`n" -ForegroundColor Yellow

if ([string]::IsNullOrWhiteSpace($increment)) {
    Write-Host "Select version increment:"
    Write-Host "  [1] Patch: v$patchVer (bugfixes, tweaks) [DEFAULT]"
    Write-Host "  [2] Minor: v$minorVer (new features)"
    Write-Host "  [3] Major: v$majorVer (breaking changes)"
    Write-Host "  [4] Custom version"
    $choice = Read-Host "`nEnter choice [1-4, or press Enter for Patch]"
    switch ($choice.Trim()) {
        "2" { $newVersion = $minorVer }
        "3" { $newVersion = $majorVer }
        "4" {
            $custom = Read-Host "Enter custom version number (e.g. 1.6.0)"
            if ([string]::IsNullOrWhiteSpace($custom)) {
                $newVersion = $patchVer
            } else {
                $newVersion = $custom.Trim().TrimStart('v')
            }
        }
        Default { $newVersion = $patchVer }
    }
} else {
    switch ($increment.ToLower()) {
        "major" { $newVersion = $majorVer }
        "minor" { $newVersion = $minorVer }
        "patch" { $newVersion = $patchVer }
        Default { $newVersion = $increment.Trim().TrimStart('v') }
    }
}

Write-Host "`nTarget version: v$newVersion" -ForegroundColor Green
$releaseNotes = Read-Host "Release notes / summary (optional, press Enter to generate from commits)"

$confirm = Read-Host "`nProceed with building and publishing v$newVersion? [Y/n]"
if (![string]::IsNullOrWhiteSpace($confirm) -and $confirm -notmatch "^[yY]") {
    Write-Host "Release cancelled by user." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nUpdating version files to $newVersion..."
Set-Content $versionFile $newVersion
(Get-Content $csprojFile) -replace '<Version>.*</Version>', "<Version>$newVersion</Version>" |
    Set-Content $csprojFile

# ===== CLEAN =====
Write-Host "Cleaning publish folder..."
if (Test-Path $publishDir) {
    Remove-Item $publishDir -Recurse -Force
}

# ===== BUILD =====
Write-Host "Building project..."
dotnet publish /p:PublishProfile=$publishProfile

# ===== POST PROCESS =====
Write-Host "Finalizing output..."

# Copy extra files into publish directory (including default configurations and update script)
$filesToCopy = @(
    $versionFile,
    "..\..\audio_parameter_web_ui.py",
    "..\..\aux_in_alsa_state.state",
    "..\..\update.sh",
    "presets.json",
    "static_settings.json"
)

foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        $destination = Join-Path $publishDir (Split-Path $file -Leaf)
        Copy-Item $file $destination -Force
    }
}

$directoriesToCopy = @(
    "..\..\web_ui"
)

foreach ($dir in $directoriesToCopy) {
    if (Test-Path $dir) {
        Copy-Item $dir $publishDir -Recurse -Force
    }
}

# Clean Python bytecode cache files before packaging
Get-ChildItem -Path $publishDir -Recurse -Include "__pycache__", "*.pyc", "*.pyo" |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Ensure update.sh has Unix LF line endings in release package
$shFile = Join-Path $publishDir "update.sh"
if (Test-Path $shFile) {
    $shContent = [System.IO.File]::ReadAllText($shFile) -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($shFile, $shContent, (New-Object System.Text.UTF8Encoding $false))
}

# ===== PACKAGE =====
Write-Host "Creating archive with cross-platform forward-slash paths..."
$zipName = "app.zip"
$zipPath = Join-Path $scriptDir $zipName

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$basePublishPath = (Resolve-Path $publishDir).Path
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)

Get-ChildItem -Path $basePublishPath -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($basePublishPath.Length).TrimStart('\', '/').Replace('\', '/')
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, $rel) | Out-Null
}
$zip.Dispose()

Write-Host "Creating GitHub release v$newVersion..."
if ([string]::IsNullOrWhiteSpace($releaseNotes)) {
    gh release create "v$newVersion" $zipName --title "v$newVersion" --generate-notes
} else {
    gh release create "v$newVersion" $zipName --title "v$newVersion" --notes "$releaseNotes"
}
Write-Host "Release v$newVersion created successfully!" -ForegroundColor Green

Write-Host "Cleanup temporary files..."
if (Test-Path $zipName) {
    Remove-Item $zipName -Force
}

if (Test-Path $publishDir) {
    Remove-Item $publishDir -Recurse -Force
}

Write-Host "Committing and pushing version update..."
git add $versionFile
git add $csprojFile
git commit -m "Release v$newVersion"
git push

Write-Host "`n=== RELEASE v$newVersion COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
