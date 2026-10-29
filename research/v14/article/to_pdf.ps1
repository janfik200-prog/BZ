# DOCX -> PDF через установленный Word (только чтение исходника); печатает число страниц.
param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $pages = $doc.ComputeStatistics(2)
    $doc.SaveAs([ref]$Pdf, [ref]17)
    $doc.Close([ref]$false)
    Write-Output "pages=$pages"
} finally {
    $word.Quit()
}
