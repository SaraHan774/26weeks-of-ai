# 26주 LLM 본질 이해 로드맵

> **목표**: Claude, Copilot 같은 LLM이 어떻게 텍스트를 이해하고 생성하는지 원리 파악
> **대상**: 비전공 안드로이드 개발자 (5년차, 수학 약함)
> **시간**: 주 10시간 (출퇴근 2시간/일 + 주말 3-4시간)

---

## Phase 1: Python과 기초 체력 (1-4주)

| 주차 | 주제 | 목표 | 핵심 개념 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:-------|:----------|
| 1 | Python 환경 세팅과 기초 문법 | Kotlin 개발자가 Python에 빠르게 적응 | 동적 타입, 들여쓰기, 리스트 컴프리헨션, None 처리 | 📚 점프 투 파이썬 (박응용, 무료 온라인)<br>🎬 Corey Schafer "Python Tutorial for Beginners" | CLI 프로그램 (파일 읽고 단어 빈도수 세기) |
| 2 | NumPy 기초 | 배열 연산 감 잡기 (텐서 연산의 기초) | 배열 생성/인덱싱/슬라이싱, 브로드캐스팅, 행렬 곱셈 `@` | 🎬 freeCodeCamp "NumPy Tutorial" (1시간)<br>📄 NumPy 공식 Quickstart | 이미지를 NumPy 배열로 불러와 밝기 조절, 흑백 변환 |
| 3 | 수학 직관 잡기 | 수식을 "읽을 수 있는" 수준으로 | 벡터(방향과 크기), 행렬(변환 표현), 미분(변화율) | 🎬 3Blue1Brown "Essence of Linear Algebra" 1-4편<br>🎬 3Blue1Brown "Essence of Calculus" 1-2편 | NumPy로 벡터 덧셈, 행렬 곱셈 시각화 |
| 4 | 신경망 감 잡기 | "학습한다"는 게 뭔지 직관적 이해 | 퍼셉트론, 손실 함수, 경사하강법 | 🎬 3Blue1Brown "Neural Networks" 4편<br>📚 밑바닥부터 시작하는 딥러닝 1-3장 | 순수 NumPy로 AND, OR 게이트 학습 |

---

## Phase 2: 딥러닝 기초 (5-9주)

| 주차 | 주제 | 목표 | 핵심 개념 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:-------|:----------|
| 5 | PyTorch 입문 | 딥러닝 프레임워크 손에 익히기 | Tensor 연산, autograd(자동 미분), GPU 사용 | 📄 PyTorch 공식 "60 Minute Blitz"<br>🎬 freeCodeCamp "PyTorch for Deep Learning" 첫 2시간 | AND/OR 게이트를 PyTorch로 재구현 |
| 6 | 첫 번째 신경망 | MNIST 손글씨 분류기 완성 | DataLoader, MLP(다층 퍼셉트론), 학습 루프 (forward→loss→backward→update) | 🎬 fast.ai "Practical Deep Learning" Lesson 1-2<br>📚 밑바닥부터 시작하는 딥러닝 4-5장 | MNIST 정확도 95% 이상 달성 |
| 7 | 학습 과정 깊이 이해 | 학습이 되고/안 되는 이유 파악 | 역전파(체인룰), 과적합/정규화, 배치 크기/학습률 | 🎬 Andrej Karpathy "Backpropagation explained"<br>📚 밑바닥부터 시작하는 딥러닝 6장 | 학습률 변경하며 학습 곡선 비교 |
| 8 | CNN 맛보기 | 이미지 처리 신경망 원리 (NLP 전 배경지식) | 컨볼루션(필터로 특징 추출), 풀링(정보 압축) | 🎬 3Blue1Brown "But what is a convolution?"<br>🎬 fast.ai Lesson 3 | CNN으로 MNIST 99% 달성 |
| 9 | 중간 점검 | 1-8주 복습, 빈 곳 채우기 | 전체 개념 연결 | 📚 밑바닥부터 시작하는 딥러닝 전체 훑기<br>🔍 막히는 개념 블로그 검색 | "신경망 학습 원리" 설명글 작성 |

---

## Phase 3: 자연어 처리 기초 (10-14주)

| 주차 | 주제 | 목표 | 핵심 개념 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:-------|:----------|
| 10 | 텍스트를 숫자로 | 컴퓨터가 언어 다루는 첫 단계 | 토큰화, 어휘집(vocabulary), One-hot 인코딩 한계 | 📚 밑바닥부터 시작하는 딥러닝 2 1-2장<br>🎬 Hugging Face NLP Course Ch.1-2 | 한글 텍스트 토큰화 (띄어쓰기 vs 형태소) |
| 11 | 단어 임베딩 | "왕-남자+여자=여왕" 원리 이해 | Word2Vec, 임베딩 공간, 사전학습 임베딩 | 📚 밑바닥부터 시작하는 딥러닝 2 3-4장<br>📄 Word2Vec 논문 (개념만) | Gensim으로 한국어 Word2Vec 학습 |
| 12 | 순환 신경망 (RNN) | 순서 있는 데이터 처리 방법 | RNN의 "기억", Vanishing gradient, LSTM/GRU | 📚 밑바닥부터 시작하는 딥러닝 2 5-6장<br>🎬 StatQuest "RNN, LSTM, GRU" | LSTM으로 텍스트 생성기 |
| 13 | Seq2Seq | 문장→문장 변환 구조 | Encoder-Decoder, Teacher forcing | 📚 밑바닥부터 시작하는 딥러닝 2 7장<br>📄 "Sequence to Sequence Learning" (2014) | 영어→한글 날짜 형식 변환기 |
| 14 | Attention 등장 | Transformer 이전 혁신 이해 | Query/Key/Value, Attention 가중치 | 📚 밑바닥부터 시작하는 딥러닝 2 8장<br>📝 Jay Alammar "Visualizing Neural MT" | Attention 가중치 히트맵 시각화 |

---

## Phase 4: Transformer 깊이 파기 (15-20주)

| 주차 | 주제 | 목표 | 핵심 개념 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:-------|:----------|
| 15 | "Attention is All You Need" | 역사 바꾼 논문 이해 | Self-Attention, Multi-Head Attention, Positional Encoding | 📄 "Attention is All You Need" (2017)<br>📝 Jay Alammar "The Illustrated Transformer" ⭐필독 | 논문 Figure 1 직접 그리며 설명 |
| 16 | Transformer 구조 상세 | 블록 하나하나 코드로 이해 | Layer Normalization, Feed-Forward Network, Residual Connection | 🎬 Andrej Karpathy "Let's build GPT" 전반부<br>💻 Harvard NLP "Annotated Transformer" | Self-Attention 레이어 직접 구현 |
| 17 | Transformer 구현 (1) | 작은 Transformer 밑바닥부터 | Embedding, Positional Encoding, Multi-Head Attention 구현 | 🎬 Andrej Karpathy "Let's build GPT" 계속<br>💻 nanoGPT 저장소 | PyTorch로 Attention 블록 완성 |
| 18 | Transformer 구현 (2) | 전체 모델 완성 및 학습 | Transformer 블록 쌓기, 학습 루프 | 💻 nanoGPT 전체 따라가기<br>📊 tiny Shakespeare 데이터셋 | 셰익스피어 스타일 텍스트 생성 |
| 19 | GPT vs BERT | Decoder-only vs Encoder-only | GPT(왼→오, 생성), BERT(양방향, 이해) | 📄 BERT 논문 (개념만)<br>📝 Jay Alammar "The Illustrated BERT" | Hugging Face로 BERT, GPT-2 비교 |
| 20 | 중간 복습 | Transformer 완전 정복 확인 | 전체 아키텍처 연결 | 💻 nanoGPT 처음부터 다시<br>🌐 커뮤니티 질문 활용 | "Transformer 작동 원리" 10분 발표자료 |

---

## Phase 5: 현대 LLM의 비밀 (21-26주)

| 주차 | 주제 | 목표 | 핵심 개념 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:-------|:----------|
| 21 | 스케일링의 마법 | 모델 크기와 성능의 관계 | Scaling Laws, Emergent Abilities, Chinchilla 교훈 | 📄 "Scaling Laws for Neural LM" (OpenAI)<br>📄 "Training Compute-Optimal LLM" (Chinchilla) | 모델 크기별 성능 데이터 분석 |
| 22 | 사전학습 원리 | GPT가 인터넷을 학습하는 방법 | Next Token Prediction, 데이터 정제, BPE/SentencePiece | 📄 "Language Models are Few-Shot Learners" (GPT-3)<br>🎬 Andrej Karpathy "State of GPT" | BPE 토크나이저 직접 구현 |
| 23 | RLHF | ChatGPT가 다른 이유 | SFT(좋은 예시 학습), Reward Model, PPO | 📄 "Training LMs to follow instructions" (InstructGPT)<br>📝 Hugging Face "RLHF" 블로그 | trl 라이브러리로 RLHF 맛보기 |
| 24 | In-Context Learning | 프롬프트가 작동하는 원리 | Few-shot Learning, Chain-of-Thought | 📄 "Chain-of-Thought Prompting" (Google)<br>📄 "LLMs are Zero-Shot Reasoners" | 동일 문제를 다양한 프롬프트로 실험 |
| 25 | 최신 기법들 | 현재 연구 방향 파악 | Constitutional AI, RAG, 멀티모달 | 📄 "Constitutional AI" (Anthropic)<br>📝 Anthropic 기술 블로그 | LangChain으로 RAG 시스템 구축 |
| 26 | 총정리 | 전체 여정 돌아보기 | 빈 곳 체크, 다음 방향 설정 | 📝 자신의 노트와 코드<br>🌐 AI 커뮤니티 참여 시작 | "내가 이해한 LLM" 블로그 포스트 |

---

## 핵심 리소스 요약

### 📚 필수 도서

| 순서 | 도서명 | 저자 | 해당 주차 | 비고 |
|:---:|:------|:-----|:---------|:-----|
| 1 | 밑바닥부터 시작하는 딥러닝 | 사이토 고키 | 4-9주 | 신경망 기초 |
| 2 | 밑바닥부터 시작하는 딥러닝 2 | 사이토 고키 | 10-14주 | 자연어 처리 |
| 3 | 점프 투 파이썬 | 박응용 | 1주 | 무료 온라인 |

### 🎬 핵심 영상 강의

| 강의명 | 채널/플랫폼 | 해당 주차 | 특징 |
|:------|:-----------|:---------|:-----|
| Essence of Linear Algebra | 3Blue1Brown | 3주 | 수학 직관 (한글 자막) |
| Neural Networks | 3Blue1Brown | 4주 | 신경망 시각화 |
| Practical Deep Learning | fast.ai | 6, 8주 | 실전 중심 |
| Let's build GPT | Andrej Karpathy | 16-18주 | Transformer 구현 |
| State of GPT | Andrej Karpathy | 22주 | 현대 LLM 개요 |

### 📝 필독 블로그

| 블로그/글 | 저자 | 해당 주차 | 특징 |
|:---------|:-----|:---------|:-----|
| The Illustrated Transformer | Jay Alammar | 15주 | ⭐ 최고의 시각적 설명 |
| The Illustrated BERT | Jay Alammar | 19주 | BERT 구조 이해 |
| Visualizing Neural MT | Jay Alammar | 14주 | Attention 시각화 |
| RLHF 설명 | Hugging Face | 23주 | 인간 피드백 학습 |

### 💻 실습 코드

| 저장소/튜토리얼 | 해당 주차 | 용도 |
|:--------------|:---------|:-----|
| nanoGPT (Andrej Karpathy) | 17-18, 20주 | GPT 구현 |
| Annotated Transformer (Harvard NLP) | 16주 | Transformer 코드 해설 |
| Hugging Face Transformers | 19, 25주 | 사전학습 모델 사용 |
| trl 라이브러리 | 23주 | RLHF 실습 |

---

## 주차별 핵심 질문

| Phase | 주차 | 이 단계를 마치면 답할 수 있어야 하는 질문 |
|:------|:---:|:----------------------------------------|
| 1 | 4주 | "컴퓨터가 어떻게 숫자들의 패턴을 스스로 찾아내지?" |
| 2 | 9주 | "신경망이 '학습'한다는 건 정확히 무슨 뜻이지?" |
| 3 | 14주 | "컴퓨터가 어떻게 '문맥'을 파악하지?" |
| 4 | 20주 | "왜 층을 깊이 쌓으면 더 복잡한 패턴을 잡아낼까?" |
| 5 | 26주 | "왜 같은 모델인데 프롬프트에 따라 결과가 달라지지?" |

---

## 학습 팁

| 카테고리 | 팁 |
|:--------|:---|
| 수학 걱정 줄이기 | 미분 = "기울기 = 얼마나 바꿔야 하는지 방향", 행렬 곱셈 = "여러 연산 한번에 묶기" |
| 안드로이드 개발자 강점 | 디버깅 능력 → 모델 학습도 "왜 안 되지?" 찾기, 아키텍처 감각 → 레이어 구조 이해 |
| 시간 배분 | 평일 출퇴근 2시간: 영상/논문, 주말 3-4시간: 코드 실습 |
| 확장 가능성 | on-device ML (TensorFlow Lite) 프로젝트로 안드로이드와 연결 |
