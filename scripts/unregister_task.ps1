# bus-data 수집 작업 스케줄러 해제
# 실행: powershell -ExecutionPolicy Bypass -File scripts\unregister_task.ps1

$TaskName = "bus-data collect"
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "삭제 완료: '$TaskName'"
