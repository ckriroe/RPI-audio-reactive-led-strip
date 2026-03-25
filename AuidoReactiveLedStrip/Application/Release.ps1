param(
    [string]$increment = "patch"  # patch | minor | major
)

$ErrorActionPreference = "Stop"

# ===== CONFIG =====
$publishDir = "bin\Release\net8.0\publish"
$publishProfile = "Properties\PublishProfiles\PublishRaspiFiles.pubxml"
$versionFile = "version.txt"
$csprojFile = "Application.csproj"

# ===== VERSIONING =====
Write-Host "Updating version..."

if (!(Test-Path $versionFile)) {
    "0.0.0" | Set-Content $versionFile
}

$version = Get-Content $versionFile
$parts = $version.Split('.')

[int]$major = $parts[0]
[int]$minor = $parts[1]
[int]$patch = $parts[2]

$oldVersion = "$major.$minor.$patch"
switch ($increment) {
    "major" { $major++; $minor = 0; $patch = 0 }
    "minor" { $minor++; $patch = 0 }
    "patch" { $patch++ }
}

$newVersion = "$major.$minor.$patch"
Write-Host "Version updated: $oldVersion -> $newVersion"

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
$filesToRemove = @(
    "dynamic_settings.json",
    "static_settings.json"
)

foreach ($file in $filesToRemove) {
    $path = Join-Path $publishDir $file
    if (Test-Path $path) {
        Remove-Item $path -Force
    }
}

$filesToCopy = @(
    "..\..\audio_parameter_web_ui.py",
    "..\..\aux_in_alsa_state.state"
)

foreach ($file in $filesToCopy) {
    $destination = Join-Path $publishDir (Split-Path $file -Leaf)
    Copy-Item $file $destination -Force
}

# ===== PACKAGE =====
Write-Host "Creating archive..."
$zipName = "app.zip"

if (Test-Path $zipName) {
    Remove-Item $zipName -Force
}

Compress-Archive -Path "$publishDir\*" -DestinationPath $zipName

Write-Host "Creating GitHub release..."
gh release create "v$newVersion" $zipName --title "v$newVersion"
Write-Host "Release v$newVersion created successfully!"

Write-Host "Cleanup..."
if (Test-Path $zipName) {
    Remove-Item $zipName -Force
}

if (Test-Path $publishDir) {
    Remove-Item $publishDir -Recurse -Force
}

Write-Host "Bump version to $newVersion..."
Set-Content $versionFile $newVersion
(Get-Content $csprojFile) -replace '<Version>.*</Version>', "<Version>$newVersion</Version>" |
        Set-Content $csprojFile

git add $versionFile
git add $csprojFile
git commit -m "Release v$newVersion"
git push

Write-Host "DONE"