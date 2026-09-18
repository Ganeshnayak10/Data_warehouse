$source = ".\data\sales"
$destination = ".\staged_sales"

Get-ChildItem $source -File -Filter "*.csv" | ForEach-Object {

    $name = $_.Name

    if ($name -match '^SALES_(S\d{2})_(\d{8})(?:__R\d+)?\.csv$') {

        $store = $matches[1]
        $date = $matches[2]

        $year = $date.Substring(0,4)
        $month = $date.Substring(4,2)

        $folder = Join-Path $destination "year=$year\month=$month\store_id=$store"

        New-Item -ItemType Directory -Force $folder | Out-Null

        Copy-Item $_.FullName -Destination $folder -Force
    }
}