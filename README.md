# 중·한 및 간체·번체 교차 언어 검색 실험

본 저장소는 다음 두 가지 주요 실험으로 구성되어 있습니다:  
1. 간체/번체 중국어 구조 분석 및 검색 실험 (test2) 
2. 중·한 교차 언어 검색 및 번역 경로 비교 실험 (test_3)

---

##  저장소 구조
```
├─ README.md                 # 프로젝트 설명 
├─ requirements.txt          # 필요한 패키지 목록
├─ test2.py                  # 실험 2: 간체·번체 네트워크 분석 및 교차 검색
├─ test_3.py                 # 실험 3: 중·한 교차 언어 검색 및 번역 경로 비교
├─ results/                  # 실험 결과 출력 파일
└─ docs/                     #  보고서 파일
```

---

##  설치 방법
필요한 패키지를 설치하려면 다음 명령어를 실행하십시오:

```bash
pip install -r requirements.txt
```

---

##  실험 1 & 2 — 간체 vs 번체 중국어 비교 (test2)

실행 방법:

```bash
python test2.py
```

주요 기능:

- 간체(CMRC) 및 번체(DRCD) JSON 데이터 로드
- corpus / queries / qrels 자동 생성
- word-level 네트워크 분석 (networkx 활용)
- 번체→간체 교차 검색 실험 수행
- Recall@5 및 네트워크 지표 출력

---

##  실험 3 — 중·한 교차 언어 검색 (test_3)

실행 방법:

```bash
python test_3.py
```

주요 기능:

- 중·한 병렬 말뭉치(ch_ko.csv) 로드
- 두 가지 번역 경로 비교:
  1. Simple Dictionary 기반 치환
  2. 영어 중개 (중국어 → 영어 → 한국어)
- SentenceTransformer + FAISS 기반 검색 수행
- MAP / P@K / nDCG 계산
- results/ 디렉토리에 검색 결과 및 평가 지표 저장

---

##  데이터

본 프로젝트는 다음 데이터를 사용합니다:

- CMRC2018 (간체 중국어 QA 데이터)
- DRCD (번체 중국어 QA 데이터)
- 중·한 병렬 데이터 ch_ko.csv

---

##  연락처

ZHAO ZEKAI  
Email:1961258171@QQ.com
