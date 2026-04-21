# Start all three services for local development (Windows).
# Run from the project root: .\scripts\dev.ps1
# Requires: .env file in project root with ABLY_API_KEY etc.

# Load .env variables into the current process
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]*)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

Write-Host "Starting FastAPI backend on http://localhost:8000 ..."
$api = Start-Process python -ArgumentList "-m", "uvicorn", "api.main:app", "--reload", "--port", "8000" -PassThru -NoNewWindow

Write-Host "Installing frontend dependencies ..."
Push-Location frontend
npm install --silent
$frontend = Start-Process npm -ArgumentList "run", "dev" -PassThru -NoNewWindow
Pop-Location

Write-Host "Starting audio inference pipeline ..."
$audio = Start-Process python -ArgumentList "-m", "inference.live_inference", "--location", "Main Entrance" -PassThru -NoNewWindow

Write-Host ""
Write-Host "Services running:"
Write-Host "  API      -> http://localhost:8000  (PID $($api.Id))"
Write-Host "  Frontend -> http://localhost:5173  (PID $($frontend.Id))"
Write-Host "  Audio    -> background             (PID $($audio.Id))"
Write-Host ""
Write-Host "Press Ctrl+C to stop all services."

try {
    Wait-Process -Id $api.Id, $frontend.Id, $audio.Id
} finally {
    Stop-Process -Id $api.Id, $frontend.Id, $audio.Id -ErrorAction SilentlyContinue
}
