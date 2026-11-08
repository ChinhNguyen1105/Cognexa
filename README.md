$files = @(Get-ChildItem .\documents\evaluation\*.txt)

if ($files.Count -eq 0) {
    throw "Không tìm thấy tài liệu .txt trong documents\evaluation"
}

Write-Host "`nCác tài liệu sẽ được nạp lại:"
$files | ForEach-Object { Write-Host " - $($_.Name)" }

$confirm = Read-Host "`nThao tác này sẽ XÓA toàn bộ dữ liệu VectorStore. Nhập RESET để xác nhận"

if ($confirm -ne "RESET") {
    Write-Host "Đã hủy thao tác."
    return
}

Write-Host "`n[1/2] Reset VectorStore..."

python -c "from cognexa.cli.assistant.commands import create_vector_store; create_vector_store().reset()"

if ($LASTEXITCODE -ne 0) {
    throw "Reset VectorStore thất bại."
}

Write-Host "`n[2/2] Nạp lại tài liệu..."

foreach ($file in $files) {
    Write-Host "`nĐang nạp: $($file.Name)"

    cognexa knowledge add $file.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "Nạp tài liệu thất bại: $($file.Name)"
    }
}

Write-Host "`nHoàn tất nạp lại dữ liệu!"