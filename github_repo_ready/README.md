# Maple Guild OCR Tool

메이플스토리 길드 스크린샷에서 길드원 닉네임 crop, 수로 점수(F), 플래그 점수(G)를 추출하는 PySide6 기반 도구입니다.

현재 버전 특징:
- 공식 길드원 명단 수집
- 여러 이미지 배치 처리
- 닉네임 crop 저장 및 OCR 보정
- F/G 점수 추출
- 결과를 Excel(.xlsx)로 저장
- 실제 닉네임 열(E열) 색상 규칙
  - `match_score >= 0.75`: 연한 보라색
  - `match_score < 0.35`: 연한 회색

## 파일
- `maple_guild_ocr.py`: 실행용 메인 스크립트
- `final_batch_queue_koreanv5_fgsplit_hybridmatch_fillrules.py`: 원본 파일명 유지본

## 실행
```bash
python maple_guild_ocr.py
```

## 필요 라이브러리
```bash
pip install -r requirements.txt
```

## 주의
- PaddleOCR 모델 다운로드에 시간이 걸릴 수 있습니다.
- Windows에서는 Python과 Visual C++ 런타임 환경에 따라 설치 이슈가 있을 수 있습니다.
- 실행 결과 Excel 파일과 임시 crop 이미지가 생성될 수 있습니다.
