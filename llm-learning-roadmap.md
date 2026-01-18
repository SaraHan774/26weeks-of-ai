# 26주 LLM 본질 이해 로드맵

> **목표**: Claude, Copilot 같은 LLM이 어떻게 텍스트를 이해하고 생성하는지 원리 파악
> **대상**: 비전공 안드로이드 개발자 (5년차, 수학 약함)
> **시간**: 주 10시간 (출퇴근 2시간/일 + 주말 3-4시간)

---

## Phase 1: Python과 기초 체력 (1-4주)

| 주차 | 주제 | 목표 | 핵심 개념 | 이론 심화 (2026) | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------------|:-------|:----------|
| 1 | Python 환경 세팅과 기초 문법 | Kotlin 개발자가 Python에 빠르게 적응 | 동적 타입, 들여쓰기, 리스트 컴프리헨션, None 처리 | **구조화 데이터 설계**: JSON/YAML로 AI 프롬프트용 데이터 구조 설계하기 | 📚 점프 투 파이썬 (박응용, 무료 온라인)<br>🎬 Corey Schafer "Python Tutorial for Beginners" | CLI 프로그램 (파일 읽고 단어 빈도수 세기) + JSON 출력 |
| 2 | NumPy 기초 | 배열 연산 감 잡기 (텐서 연산의 기초) | 배열 생성/인덱싱/슬라이싱, 브로드캐스팅, 행렬 곱셈 `@` | **확률 분포의 직관**: NumPy로 정규분포, 균등분포 시각화 | 🎬 freeCodeCamp "NumPy Tutorial" (1시간)<br>📄 NumPy 공식 Quickstart | 이미지를 NumPy 배열로 불러와 밝기 조절, 흑백 변환 |
| 3 | 수학 직관 잡기 | 수식을 "읽을 수 있는" 수준으로 | 벡터(방향과 크기), 행렬(변환 표현), 미분(변화율) | **선형대수의 기하학**: 벡터 공간에서 변환이 의미하는 것 | 🎬 3Blue1Brown "Essence of Linear Algebra" 1-4편<br>🎬 3Blue1Brown "Essence of Calculus" 1-2편 | NumPy로 벡터 덧셈, 행렬 곱셈 시각화 |
| 4 | 신경망 감 잡기 | "학습한다"는 게 뭔지 직관적 이해 | 퍼셉트론, 손실 함수, 경사하강법 | **엔트로피와 교차 엔트로피**: 모델이 "정보"를 어떻게 측정하나? "놀람의 크기"를 줄이는 학습의 원리 | 🎬 3Blue1Brown "Neural Networks" 4편<br>📚 밑바닥부터 시작하는 딥러닝 1-3장<br>🎬 StatQuest "Entropy" | 순수 NumPy로 AND, OR 게이트 학습 + 손실 함수 시각화 |

---

## Phase 2: 딥러닝 기초 (5-9주)

| 주차 | 주제 | 목표 | 핵심 개념 | 이론 심화 (2026) | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------------|:-------|:----------|
| 5 | PyTorch 입문 | 딥러닝 프레임워크 손에 익히기 | Tensor 연산, autograd(자동 미분), GPU 사용 | **자동 미분의 원리**: Computational Graph와 역전파의 관계 | 📄 PyTorch 공식 "60 Minute Blitz"<br>🎬 freeCodeCamp "PyTorch for Deep Learning" 첫 2시간 | AND/OR 게이트를 PyTorch로 재구현 |
| 6 | 첫 번째 신경망 | MNIST 손글씨 분류기 완성 | DataLoader, MLP(다층 퍼셉트론), 학습 루프 (forward→loss→backward→update) | **Softmax와 확률 분포**: 왜 마지막 층에 Softmax를 쓰는가? | 🎬 fast.ai "Practical Deep Learning" Lesson 1-2<br>📚 밑바닥부터 시작하는 딥러닝 4-5장 | MNIST 정확도 95% 이상 달성 |
| 7 | 학습 과정 깊이 이해 | 학습이 되고/안 되는 이유 파악 | 역전파(체인룰), 과적합/정규화, 배치 크기/학습률 | **정보 이론 관점의 손실**: Cross-Entropy Loss가 왜 분류 문제에 최적인가? | 🎬 Andrej Karpathy "Backpropagation explained"<br>📚 밑바닥부터 시작하는 딥러닝 6장 | 학습률 변경하며 학습 곡선 비교 + 다양한 손실 함수 실험 |
| 8 | CNN 맛보기 | 이미지 처리 신경망 원리 (NLP 전 배경지식) | 컨볼루션(필터로 특징 추출), 풀링(정보 압축) | **컨볼루션의 수학적 의미**: 신호 처리 관점에서의 컨볼루션 | 🎬 3Blue1Brown "But what is a convolution?"<br>🎬 fast.ai Lesson 3 | CNN으로 MNIST 99% 달성 |
| 9 | 중간 점검 | 1-8주 복습, 빈 곳 채우기 | 전체 개념 연결 | **경사하강법의 변형들**: SGD, Adam, AdamW 비교 | 📚 밑바닥부터 시작하는 딥러닝 전체 훑기<br>🔍 막히는 개념 블로그 검색 | "신경망 학습 원리" 설명글 작성 |

---

## Phase 3: 자연어 처리 기초 (10-14주)

| 주차 | 주제 | 목표 | 핵심 개념 | 이론 심화 (2026) | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------------|:-------|:----------|
| 10 | 텍스트를 숫자로 | 컴퓨터가 언어 다루는 첫 단계 | 토큰화, 어휘집(vocabulary), One-hot 인코딩 한계 | **토큰화 알고리즘 심화**: 한글이 왜 영어보다 토큰을 많이 먹는지, BPE vs WordPiece | 📚 밑바닥부터 시작하는 딥러닝 2 1-2장<br>🎬 Hugging Face NLP Course Ch.1-2 | 한글 텍스트 토큰화 (띄어쓰기 vs 형태소) |
| 11 | 단어 임베딩 | "왕-남자+여자=여왕" 원리 이해 | Word2Vec, 임베딩 공간, 사전학습 임베딩 | **벡터 공간의 기하학**: 단어들의 "의미 거리"를 어떻게 측정하나? | 📚 밑바닥부터 시작하는 딥러닝 2 3-4장<br>📄 Word2Vec 논문 (개념만) | Gensim으로 한국어 Word2Vec 학습 |
| 12 | 순환 신경망 (RNN) | 순서 있는 데이터 처리 방법 | RNN의 "기억", Vanishing gradient, LSTM/GRU | **시퀀스 모델링의 이론**: 왜 RNN은 긴 문장을 잘 못 외우나? Gating의 수학적 원리 | 📚 밑바닥부터 시작하는 딥러닝 2 5-6장<br>🎬 StatQuest "RNN, LSTM, GRU" | LSTM으로 텍스트 생성기 |
| 13 | Seq2Seq | 문장→문장 변환 구조 | Encoder-Decoder, Teacher forcing | **정보 압축의 병목**: 고정 길이 벡터의 한계와 Attention의 필요성 | 📚 밑바닥부터 시작하는 딥러닝 2 7장<br>📄 "Sequence to Sequence Learning" (2014) | 영어→한글 날짜 형식 변환기 |
| 14 | Attention 등장 | Transformer 이전 혁신 이해 | Query/Key/Value, Attention 가중치 | **코사인 유사도와 검색**: Dense vs Sparse 벡터, Attention Score 계산의 의미 | 📚 밑바닥부터 시작하는 딥러닝 2 8장<br>📝 Jay Alammar "Visualizing Neural MT" | Attention 가중치 히트맵 시각화 |

---

### 🎨 Phase 3.5: 멀티모달 입문 (14.5주)

**2026년 필수 추가**: 텍스트만 다루는 시대는 지났습니다. 이미지, 음성 데이터를 함께 처리하는 능력이 기본이 되었습니다.

| 주차 | 주제 | 목표 | 핵심 개념 | 2026년 실무 포인트 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------------|:-------|:----------|
| 14.5 | 멀티모달 워크플로우 | 텍스트 외 데이터(이미지, 음성) 처리 흐름 이해 | Vision Transformer (ViT) 개념, CLIP (이미지-텍스트 연결), OCR 기초 | **멀티모달 파이프라인**: 영수증 사진 → 텍스트 추출 → 데이터 변환의 전체 흐름 | 🎬 Hugging Face "Multimodal Models"<br>📄 CLIP 논문 (개념만)<br>💻 EasyOCR 라이브러리 | 영수증 사진을 찍어 텍스트 추출 후 가계부 JSON 데이터로 변환 |

---

## Phase 4: Transformer 깊이 파기 (15-20주)

| 주차 | 주제 | 목표 | 핵심 개념 | 이론 심화 & 2026년 업데이트 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:------------------------|:-------|:----------|
| 15 | "Attention is All You Need" + 추론 모델 | 역사 바꾼 논문 이해 + 2026년 진화 | Self-Attention, Multi-Head Attention, Positional Encoding | **추론 모델의 이해**: System 1 (빠른 직관) vs System 2 (느린 추론) 사고, Chain-of-Thought의 자동화 (OpenAI o1, DeepSeek-R1) | 📄 "Attention is All You Need" (2017)<br>📝 Jay Alammar "The Illustrated Transformer" ⭐필독<br>📄 OpenAI o1 System Card | 논문 Figure 1 직접 그리며 설명 + o1으로 복잡한 알고리즘 문제 풀고 사고 로그 분석 |
| 16 | Transformer 구조 상세 | 블록 하나하나 코드로 이해 | Layer Normalization, Feed-Forward Network, Residual Connection | **Scaling Laws와 정규화**: 모델이 커질수록 왜 똑똑해지나? Layer Norm, Residual Connection의 수학적 의의 | 🎬 Andrej Karpathy "Let's build GPT" 전반부<br>💻 Harvard NLP "Annotated Transformer" | Self-Attention 레이어 직접 구현 |
| 17 | Transformer 구현 (1) | 작은 Transformer 밑바닥부터 | Embedding, Positional Encoding, Multi-Head Attention 구현 | **Attention Score의 의미**: Softmax와 확률 분포, Temperature의 역할 | 🎬 Andrej Karpathy "Let's build GPT" 계속<br>💻 nanoGPT 저장소 | PyTorch로 Attention 블록 완성 |
| 18 | Transformer 구현 (2) + GraphRAG | 전체 모델 완성 및 학습 | Transformer 블록 쌓기, 학습 루프 | **GraphRAG**: 단순 검색이 아닌 지식 그래프를 활용한 복잡한 관계 추론 | 💻 nanoGPT 전체 따라가기<br>📊 tiny Shakespeare 데이터셋<br>📄 GraphRAG 논문 (MS Research) | 셰익스피어 스타일 텍스트 생성 |
| 19 | GPT vs BERT + SLM 최적화 | Decoder-only vs Encoder-only + 경량화 | GPT(왼→오, 생성), BERT(양방향, 이해) | **SLM(Small Language Model) 최적화**: 양자화(Quantization), Llama 4-3B 같은 경량 모델 로컬 실행, 내 컴퓨터에서 AI 돌리기 | 📄 BERT 논문 (개념만)<br>📝 Jay Alammar "The Illustrated BERT"<br>💻 Ollama 공식 문서 | Hugging Face로 BERT, GPT-2 비교 + Ollama로 Llama 3 로컬 실행 및 양자화 실험 |
| 20 | 중간 복습 + Alignment 이론 | Transformer 완전 정복 확인 | 전체 아키텍처 연결 | **정렬(Alignment)의 이론**: RLHF에서 DPO(Direct Preference Optimization)까지, 모델이 인간의 의도를 학습하는 방법 | 💻 nanoGPT 처음부터 다시<br>📄 DPO 논문 (2023)<br>🌐 커뮤니티 질문 활용 | "Transformer 작동 원리" 10분 발표자료 |

---

## Phase 5: 현대 LLM과 실무 역량 (21-26주)

**2026년 핵심 변화**: 이제 코드는 AI가 짭니다. 개발자의 핵심 역량은 **"AI가 코드를 짤 수 있도록 시스템을 설계하고, 그 결과를 검증하는 능력"**으로 이동했습니다.

| 주차 | 주제 | 목표 | 핵심 개념 | 2026년 실무 역량 | 리소스 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------------|:-------|:----------|
| 21 | 스케일링의 마법 + AI 시스템 설계 | 모델 크기와 성능의 관계 + 설계 능력 | Scaling Laws, Emergent Abilities, Chinchilla 교훈 | **AI 중심 시스템 설계**: 마이크로서비스 아키텍처, AI가 한 번에 코딩할 수 있도록 설계 분해하기 | 📄 "Scaling Laws for Neural LM" (OpenAI)<br>📄 "Training Compute-Optimal LLM" (Chinchilla) | 모델 크기별 성능 데이터 분석 + 복잡한 앱을 AI가 코딩 가능한 Specification으로 분해 |
| 22 | 사전학습 원리 + 토큰 최적화 | GPT가 인터넷을 학습하는 방법 + 비용 절감 | Next Token Prediction, 데이터 정제, BPE/SentencePiece | **토큰화 심화**: 한글이 왜 토큰을 많이 먹는지, BPE 알고리즘 이해로 비용 절감하기 | 📄 "Language Models are Few-Shot Learners" (GPT-3)<br>🎬 Andrej Karpathy "State of GPT" | BPE 토크나이저 직접 구현 (한글 최적화) |
| 23 | RLHF + DPO | ChatGPT가 다른 이유 + 최신 정렬 기법 | SFT(좋은 예시 학습), Reward Model, PPO, **DPO (Direct Preference Optimization)** | **정렬의 실무**: 두 답변을 비교하게 함으로써 가치관 이식하기 | 📄 "Training LMs to follow instructions" (InstructGPT)<br>📝 Hugging Face "RLHF" 블로그<br>📄 DPO 논문 (2023) | trl 라이브러리로 RLHF + DPO 비교 실험 |
| 24 | In-Context Learning + Multi-Agent | 프롬프트가 작동하는 원리 + 에이전트 협업 | Few-shot Learning, Chain-of-Thought | **Agentic Workflows**: 여러 AI 에이전트가 협업하는 구조 (Multi-Agent), 리서처 에이전트 + 작가 에이전트 | 📄 "Chain-of-Thought Prompting" (Google)<br>📄 "LLMs are Zero-Shot Reasoners"<br>💻 LangGraph 튜토리얼 | 동일 문제를 다양한 프롬프트로 실험 + LangGraph로 Multi-Agent 시스템 구축 |
| 25 | RAG 심화 + MCP 서버 구축 | 현재 연구 방향 파악 + 컨텍스트 엔지니어링 | Constitutional AI, RAG, 멀티모달 | **MCP(Model Context Protocol) 서버 구축**: AI가 로컬 파일/DB를 읽는 통로 만들기, 컨텍스트 엔지니어링 | 📄 "Constitutional AI" (Anthropic)<br>📝 Anthropic 기술 블로그<br>💻 MCP 명세 (Anthropic GitHub) | LangChain으로 RAG 시스템 구축 + 내 SQLite DB와 연결된 MCP 서버 구축 |
| 26 | LLM-as-a-Judge + 총정리 | 전체 여정 돌아보기 + 자동화된 평가 | 빈 곳 체크, 다음 방향 설정 | **자동화된 평가 및 검증**: 사람 대신 더 똑똑한 모델(GPT-5급)로 내 모델 성능 자동 평가, 테스트 자동화 | 📝 자신의 노트와 코드<br>📄 LLM-as-a-Judge 논문 (2023)<br>🌐 AI 커뮤니티 참여 시작 | "내가 이해한 LLM" 블로그 포스트 + AI가 작성한 코드의 엣지 케이스 찾는 테스트 자동화 |

---

## 🆕 2026년 핵심 업데이트 요약

### Phase별 2026년 필수 추가 항목

| Phase | 주차 | 기존 주제 | 2026년 필수 추가 항목 |
|:-----:|:---:|:---------|:-------------------|
| **1-2** | 1-9주 | Python & 딥러닝 기초 | 구조화 데이터 설계 (JSON/YAML), 정보 이론 (엔트로피, Cross-Entropy) |
| **3** | 10-14주 | 자연어 처리 기초 | 토큰화 심화 (BPE 알고리즘), 벡터 공간의 기하학 |
| **3.5** | 14.5주 | **멀티모달 입문** | Vision Transformer, CLIP, OCR 파이프라인 |
| **4** | 15-16주 | Transformer | **추론 모델** (o1, DeepSeek-R1), **GraphRAG** |
| **4** | 19-20주 | GPT vs BERT | **SLM 최적화** (양자화, Ollama), **DPO** (정렬 이론) |
| **5** | 21주 | Scaling Laws | **AI 중심 시스템 설계** (마이크로서비스) |
| **5** | 23-24주 | RLHF + In-Context | **DPO**, **Multi-Agent 시스템** (Agentic Workflows) |
| **5** | 25주 | RAG | **MCP 서버 구축** (컨텍스트 엔지니어링) |
| **5** | 26주 | 총정리 | **LLM-as-a-Judge** (자동화된 평가) |

### 주요 키워드 설명

| 키워드 | 설명 | 왜 중요한가 |
|:------|:-----|:----------|
| **추론 모델** (Reasoning Models) | o1, o3, DeepSeek-R1 등 "생각하는 시간"을 갖는 모델 | 복잡한 문제 해결 시 사고 과정을 추적하여 신뢰성 검증 가능 |
| **MCP** (Model Context Protocol) | AI가 로컬 파일/DB를 안전하게 읽고 쓸 수 있는 표준화된 통로 | AI에게 나만의 데이터를 제공하는 개발자 필수 스킬 |
| **멀티모달** (Multimodal) | 텍스트+이미지+음성을 함께 처리하는 능력 | 2026년 AI 기본 사양, 영수증 OCR 등 실무 필수 |
| **SLM** (Small Language Model) | 로컬에서 돌아가는 경량 모델 (Llama 4-3B 등) | 비용 절감, 보안, 오프라인 활용 가능 |
| **DPO** (Direct Preference Optimization) | RLHF보다 효율적인 정렬 기법 | 모델 정렬의 최신 표준 |
| **Multi-Agent** | 여러 AI 에이전트가 협업하는 시스템 | 복잡한 작업을 역할별로 분담 |
| **LLM-as-a-Judge** | AI로 AI를 평가하는 자동화 시스템 | 수동 평가 대신 자동화된 품질 관리 |

---

## 🎯 Phase별 실무 역량 체크리스트

**2026년 개발자에게 요구되는 역량**: 코딩 능력보다 **시스템 설계 + 검증 능력**

| Phase | 이론 목표 | 실습 목표 | **실무 역량 목표 (2026)** |
|:-----:|:----------|:----------|:----------------------|
| **Phase 1-2**<br>(1-9주) | 신경망 학습 원리 이해 | MNIST 99% 달성 | ✅ **구조화 데이터 설계**: JSON/YAML로 AI 프롬프트용 데이터 설계<br>✅ **정보 측정**: 손실 함수를 통한 학습 이해 |
| **Phase 3**<br>(10-14주) | 문맥 파악 원리 이해 | LSTM 텍스트 생성기 | ✅ **토큰 비용 최적화**: 한글 토큰화 이해로 API 비용 절감<br>✅ **벡터 검색**: 유사도 계산으로 검색 품질 향상 |
| **Phase 3.5**<br>(14.5주) | 멀티모달 데이터 처리 이해 | 영수증 OCR 파이프라인 | ✅ **멀티모달 파이프라인**: 이미지→텍스트→데이터 전체 흐름 구축 |
| **Phase 4**<br>(15-20주) | Transformer 구조 완전 이해 | GPT 밑바닥부터 구현 | ✅ **추론 모델 활용**: 복잡한 문제에 o1 사용하고 사고 과정 검증<br>✅ **로컬 AI 운영**: Ollama로 SLM 실행 및 양자화<br>✅ **Tool Calling**: MCP 기반 도구 연동 이론 |
| **Phase 5**<br>(21-26주) | Scaling Laws, RLHF, In-Context Learning | RAG 시스템 구축 | ✅ **AI 시스템 설계**: 복잡한 앱을 AI가 코딩 가능한 단위로 분해<br>✅ **MCP 서버 구축**: AI가 내 DB를 읽는 통로 만들기<br>✅ **Multi-Agent 설계**: 여러 에이전트 협업 시스템<br>✅ **자동화된 검증**: LLM-as-a-Judge로 테스트 자동화 |

### 실무 역량 달성 기준

각 Phase를 마치면 다음을 할 수 있어야 합니다:

- **Phase 1-2 완료 시**: "AI에게 명확한 입력 데이터를 JSON으로 설계할 수 있다"
- **Phase 3 완료 시**: "토큰 수를 계산하여 API 비용을 예측할 수 있다"
- **Phase 3.5 완료 시**: "이미지를 입력받아 텍스트로 변환하는 파이프라인을 만들 수 있다"
- **Phase 4 완료 시**: "로컬에서 경량 모델을 실행하고 API와 비용/성능을 비교할 수 있다"
- **Phase 5 완료 시**: "AI가 내 DB를 읽고 쓸 수 있는 MCP 서버를 구축할 수 있다"

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
| **Ollama** | **19주** | **로컬 SLM 실행** |
| **LangGraph** | **24주** | **Multi-Agent 시스템 구축** |
| **MCP SDK** | **25주** | **Model Context Protocol 서버** |

### 📄 2026년 필수 논문 및 문서

| 논문/문서 | 저자/출처 | 해당 주차 | 핵심 내용 |
|:---------|:---------|:---------|:---------|
| OpenAI o1 System Card | OpenAI | 15주 | 추론 모델의 작동 원리 |
| DeepSeek-R1 Technical Report | DeepSeek | 15주 | 오픈소스 추론 모델 |
| DPO 논문 (Direct Preference Optimization) | Stanford | 20, 23주 | RLHF 대체 기법 |
| GraphRAG 논문 | Microsoft Research | 18주 | 지식 그래프 기반 RAG |
| LLM-as-a-Judge 논문 | Various | 26주 | AI 자동 평가 시스템 |
| CLIP 논문 (Contrastive Language-Image Pre-training) | OpenAI | 14.5주 | 멀티모달 임베딩 |
| MCP 명세 | Anthropic GitHub | 25주 | AI와 도구 연동 표준 |

### 🛠️ 2026년 필수 도구

| 도구 | 용도 | 해당 주차 | 공식 사이트 |
|:----|:-----|:---------|:-----------|
| **Ollama** | 로컬 SLM 실행 | 19주 | ollama.ai |
| **LangChain** | RAG 시스템 구축 | 25주 | python.langchain.com |
| **LangGraph** | Multi-Agent 워크플로우 | 24주 | langchain-ai.github.io/langgraph |
| **EasyOCR** | 이미지 텍스트 추출 | 14.5주 | github.com/JaidedAI/EasyOCR |
| **trl** | RLHF & DPO 학습 | 23주 | huggingface.co/docs/trl |

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

---

## 💡 비전공 개발자를 위한 2026년 최종 조언

### 1. "수학보다 시스템"
2026년의 AI 개발은 모델 내부 수식보다 **'어떤 모델과 어떤 DB를 어떻게 연결할 것인가'** 하는 아키텍처 설계 능력이 훨씬 중요합니다. 수학은 "이해의 도구"로 활용하되, 완벽한 증명까지 추구할 필요는 없습니다.

### 2. "SLM에 주목하세요"
모든 것을 거대한 클라우드 모델(OpenAI 등)에 의존하기보다, 특정 작업에 특화된 가벼운 모델(Small Language Model)을 로컬에 띄워 쓰는 기술이 개발자로서 큰 자산이 됩니다. Ollama로 Llama 3를 로컬에서 실행해보세요.

### 3. "추론의 비용을 계산하세요"
똑똑한 추론 모델(o1, o3)은 비싸고 느립니다. 어떤 상황에 '일반 모델'을 쓰고 어떤 상황에 '추론 모델'을 쓸지 결정하는 비즈니스 로직 설계자가 되세요. 예: 간단한 텍스트 요약은 GPT-4o-mini, 복잡한 알고리즘 문제는 o1.

### 4. "검증이 핵심"
Claude와 Copilot이 발전할수록 **"어떻게 코딩하는가(How)"**보다는 **"무엇을 만들고 왜 그렇게 만들어야 하는가(What & Why)"**를 고민하는 개발자가 살아남습니다. AI가 짠 코드를 테스트하고 평가하는 능력이 미래 개발자의 핵심입니다.

### 5. "이론은 디버깅 도구"
수학적 이론은 **"AI가 왜 이런 답변을 내놓았는지 추론하는 도구"**로 사용하세요. 예를 들어, 확률적 샘플링(Temperature) 개념을 알면 AI의 답변이 왜 매번 달라지는지 이해하고 제어할 수 있게 됩니다.

### 6. "Proof보다 Intuition"
수학 책의 증명 과정을 다 따라가려 하지 마세요. 대신 **"이 수식이 물리적으로 무엇을 의미하는가?"**를 설명해 주는 유튜브 강의(예: 3Blue1Brown)나 기술 블로그를 먼저 보세요.

### 7. "직접 시각화"
이론으로 배운 내용을 `matplotlib`이나 `seaborn` 라이브러리로 직접 그려보세요. 숫자가 움직이는 것을 눈으로 보면 수학적 장벽이 무너집니다.
