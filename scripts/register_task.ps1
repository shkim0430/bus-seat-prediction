# bus-data 수집 작업 스케줄러 등록
# 매일 16:00~23:00, 2분 간격으로 collect.py 를 창 없이(pythonw) 실행.
# 실행: powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1 [-PythonW <pythonw.exe 경로>]
# (참고: 현재 운영은 Linux VM의 cron. 이 스크립트는 초기 Windows 노트북 운영용.)
param([string]$PythonW = "")

$ErrorActionPreference = "Stop"

$TaskName = "bus-data collect"
if (-not $PythonW) { $PythonW = (Get-Command pythonw.exe -ErrorAction Stop).Source }
# 한글 경로 인코딩 문제 회피: 문자열 리터럴이 아니라 이 스크립트의 실제 위치에서 유도
$Project  = Split-Path -Parent $PSScriptRoot   # 저장소 루트 (scripts 의 상위)
$Script   = Join-Path $Project "collect.py"

# 창 없이 실행 (pythonw = 콘솔 없는 파이썬)
$action = New-ScheduledTaskAction -Execute $PythonW -Argument "`"$Script`"" -WorkingDirectory $Project

# 매일 16:00 시작 트리거에 2분 반복(7시간=23:00까지) 부여
# (Daily 트리거는 반복을 직접 못 받으므로 Once 트리거의 Repetition 을 빌려온다)
$trigger = New-ScheduledTaskTrigger -Daily -At 16:00
$rep = (New-ScheduledTaskTrigger -Once -At 16:00 `
        -RepetitionInterval (New-TimeSpan -Minutes 2) `
        -RepetitionDuration (New-TimeSpan -Hours 7)).Repetition
$trigger.Repetition = $rep

# 겹침 방지(이전 실행 안 끝났으면 새로 안 띄움) + 늦어도 실행 + 3분 넘으면 강제 종료
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 3) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

# 로그온한 현재 사용자로 실행 (비밀번호 저장 불필요)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "M6117 등 김포행 광역버스 도착·좌석 데이터 수집 (16:00-23:00, 2분 간격)" `
    -Force | Out-Null

Write-Host "등록 완료: '$TaskName'"
Write-Host "  실행:   $PythonW `"$Script`""
Write-Host "  트리거: 매일 16:00~23:00, 2분 간격"
