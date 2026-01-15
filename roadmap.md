---
layout: default
title: Learning Roadmap
---

# 26주 LLM 학습 로드맵

> **목표**: Claude, Copilot 같은 LLM이 어떻게 텍스트를 이해하고 생성하는지 원리 파악
> **대상**: 비전공 안드로이드 개발자 (5년차, 수학 약함)
> **시간**: 주 10시간 (출퇴근 2시간/일 + 주말 3-4시간)

---

## 📊 학습 여정 시각화

<div class="mermaid">
graph TD
    Start([26주 LLM 학습 시작]) --> Phase1

    Phase1[Phase 1: Python과 기초 체력<br/>Week 1-4]
    Phase1 --> W1[Week 1: Python 기초]
    Phase1 --> W2[Week 2: NumPy]
    Phase1 --> W3[Week 3: 수학 직관]
    Phase1 --> W4[Week 4: 신경망 개념]

    W4 --> Phase2[Phase 2: 딥러닝 기초<br/>Week 5-9]
    Phase2 --> W5[Week 5: PyTorch]
    Phase2 --> W6[Week 6: MNIST 분류기]
    Phase2 --> W7[Week 7: 역전파]
    Phase2 --> W8[Week 8: CNN]
    Phase2 --> W9[Week 9: 중간 점검]

    W9 --> Phase3[Phase 3: 자연어 처리<br/>Week 10-14]
    Phase3 --> W10[Week 10: 토큰화]
    Phase3 --> W11[Week 11: Word2Vec]
    Phase3 --> W12[Week 12: RNN/LSTM]
    Phase3 --> W13[Week 13: Seq2Seq]
    Phase3 --> W14[Week 14: Attention]

    W14 --> Phase4[Phase 4: Transformer<br/>Week 15-20]
    Phase4 --> W15[Week 15: Attention is All You Need]
    Phase4 --> W16[Week 16: Transformer 구조]
    Phase4 --> W17[Week 17-18: 구현]
    Phase4 --> W19[Week 19: GPT vs BERT]
    Phase4 --> W20[Week 20: 중간 복습]

    W20 --> Phase5[Phase 5: 현대 LLM<br/>Week 21-26]
    Phase5 --> W21[Week 21: Scaling Laws]
    Phase5 --> W22[Week 22: 사전학습]
    Phase5 --> W23[Week 23: RLHF]
    Phase5 --> W24[Week 24: In-Context Learning]
    Phase5 --> W25[Week 25: 최신 기법]
    Phase5 --> W26[Week 26: 총정리]

    W26 --> End([LLM 원리 이해 완료!])

    style Phase1 fill:#e1f5ff,stroke:#0366d6,stroke-width:3px
    style Phase2 fill:#fff5e1,stroke:#fb8500,stroke-width:3px
    style Phase3 fill:#f0e1ff,stroke:#9d4edd,stroke-width:3px
    style Phase4 fill:#e1ffe1,stroke:#2d6a4f,stroke-width:3px
    style Phase5 fill:#ffe1e1,stroke:#e63946,stroke-width:3px
    style Start fill:#f0f0f0,stroke:#333,stroke-width:2px
    style End fill:#90EE90,stroke:#333,stroke-width:3px
</div>

<div style="margin: 32px 0; padding: 24px; background-color: #f6f8fa; border-radius: 8px; border-left: 4px solid #0366d6;">
  <strong>💡 학습 흐름:</strong> Python 기초 → 딥러닝 원리 → NLP 기초 → Transformer 구조 → 현대 LLM 이해
</div>

---

## Phase 1: Python과 기초 체력 (1-4주)

| 주차 | 주제 | 목표 | 핵심 개념 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------|
| 1 | Python 환경 세팅과 기초 문법 | Kotlin 개발자가 Python에 빠르게 적응 | 동적 타입, 들여쓰기, 리스트 컴프리헨션 | CLI 프로그램 (파일 읽고 단어 빈도수 세기) |
| 2 | NumPy 기초 | 배열 연산 감 잡기 (텐서 연산의 기초) | 배열 생성/인덱싱/슬라이싱, 브로드캐스팅, 행렬 곱셈 | 이미지를 NumPy 배열로 불러와 밝기 조절, 흑백 변환 |
| 3 | 수학 직관 잡기 | 수식을 "읽을 수 있는" 수준으로 | 벡터(방향과 크기), 행렬(변환 표현), 미분(변화율) | NumPy로 벡터 덧셈, 행렬 곱셈 시각화 |
| 4 | 신경망 감 잡기 | "학습한다"는 게 뭔지 직관적 이해 | 퍼셉트론, 손실 함수, 경사하강법 | 순수 NumPy로 AND, OR 게이트 학습 |

---

## Phase 2: 딥러닝 기초 (5-9주)

| 주차 | 주제 | 목표 | 핵심 개념 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------|
| 5 | PyTorch 입문 | 딥러닝 프레임워크 손에 익히기 | Tensor 연산, autograd, GPU 사용 | AND/OR 게이트를 PyTorch로 재구현 |
| 6 | 첫 번째 신경망 | MNIST 손글씨 분류기 완성 | DataLoader, MLP, 학습 루프 | MNIST 정확도 95% 이상 달성 |
| 7 | 학습 과정 깊이 이해 | 학습이 되고/안 되는 이유 파악 | 역전파, 과적합/정규화, 하이퍼파라미터 | 학습률 변경하며 학습 곡선 비교 |
| 8 | CNN 맛보기 | 이미지 처리 신경망 원리 | 컨볼루션, 풀링 | CNN으로 MNIST 99% 달성 |
| 9 | 중간 점검 | 1-8주 복습, 빈 곳 채우기 | 전체 개념 연결 | "신경망 학습 원리" 설명글 작성 |

---

## Phase 3: 자연어 처리 기초 (10-14주)

| 주차 | 주제 | 목표 | 핵심 개념 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------|
| 10 | 텍스트를 숫자로 | 컴퓨터가 언어 다루는 첫 단계 | 토큰화, 어휘집, One-hot 인코딩 | 한글 텍스트 토큰화 (띄어쓰기 vs 형태소) |
| 11 | 단어 임베딩 | "왕-남자+여자=여왕" 원리 이해 | Word2Vec, 임베딩 공간 | Gensim으로 한국어 Word2Vec 학습 |
| 12 | 순환 신경망 (RNN) | 순서 있는 데이터 처리 방법 | RNN의 "기억", LSTM/GRU | LSTM으로 텍스트 생성기 |
| 13 | Seq2Seq | 문장→문장 변환 구조 | Encoder-Decoder, Teacher forcing | 영어→한글 날짜 형식 변환기 |
| 14 | Attention 등장 | Transformer 이전 혁신 이해 | Query/Key/Value, Attention 가중치 | Attention 가중치 히트맵 시각화 |

---

## Phase 4: Transformer 깊이 파기 (15-20주)

| 주차 | 주제 | 목표 | 핵심 개념 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------|
| 15 | "Attention is All You Need" | 역사 바꾼 논문 이해 | Self-Attention, Multi-Head Attention, Positional Encoding | 논문 Figure 1 직접 그리며 설명 |
| 16 | Transformer 구조 상세 | 블록 하나하나 코드로 이해 | Layer Normalization, Feed-Forward Network, Residual Connection | Self-Attention 레이어 직접 구현 |
| 17 | Transformer 구현 (1) | 작은 Transformer 밑바닥부터 | Embedding, Positional Encoding, Multi-Head Attention 구현 | PyTorch로 Attention 블록 완성 |
| 18 | Transformer 구현 (2) | 전체 모델 완성 및 학습 | Transformer 블록 쌓기, 학습 루프 | 셰익스피어 스타일 텍스트 생성 |
| 19 | GPT vs BERT | Decoder-only vs Encoder-only | GPT(왼→오, 생성), BERT(양방향, 이해) | Hugging Face로 BERT, GPT-2 비교 |
| 20 | 중간 복습 | Transformer 완전 정복 확인 | 전체 아키텍처 연결 | "Transformer 작동 원리" 10분 발표자료 |

---

## Phase 5: 현대 LLM의 비밀 (21-26주)

| 주차 | 주제 | 목표 | 핵심 개념 | 주말 실습 |
|:---:|:-----|:-----|:----------|:----------|
| 21 | 스케일링의 마법 | 모델 크기와 성능의 관계 | Scaling Laws, Emergent Abilities, Chinchilla 교훈 | 모델 크기별 성능 데이터 분석 |
| 22 | 사전학습 원리 | GPT가 인터넷을 학습하는 방법 | Next Token Prediction, 데이터 정제, BPE/SentencePiece | BPE 토크나이저 직접 구현 |
| 23 | RLHF | ChatGPT가 다른 이유 | SFT(좋은 예시 학습), Reward Model, PPO | trl 라이브러리로 RLHF 맛보기 |
| 24 | In-Context Learning | 프롬프트가 작동하는 원리 | Few-shot Learning, Chain-of-Thought | 동일 문제를 다양한 프롬프트로 실험 |
| 25 | 최신 기법들 | 현재 연구 방향 파악 | Constitutional AI, RAG, 멀티모달 | LangChain으로 RAG 시스템 구축 |
| 26 | 총정리 | 전체 여정 돌아보기 | 빈 곳 체크, 다음 방향 설정 | "내가 이해한 LLM" 블로그 포스트 |

---

## 🎯 Phase별 핵심 개념

<div class="mermaid">
mindmap
  root((26주<br/>LLM 학습))
    Phase 1<br/>기초 체력
      Python 기초
        동적 타입
        리스트 컴프리헨션
      NumPy
        배열 연산
        브로드캐스팅
      수학 직관
        벡터/행렬
        미분
      신경망 개념
        퍼셉트론
        경사하강법
    Phase 2<br/>딥러닝
      PyTorch
        Tensor
        autograd
      신경망 학습
        역전파
        과적합
      CNN
        컨볼루션
        풀링
    Phase 3<br/>NLP 기초
      토큰화
        어휘집
        임베딩
      Word2Vec
        의미 공간
      RNN/LSTM
        순서 처리
        기억
      Attention
        Query/Key/Value
    Phase 4<br/>Transformer
      Self-Attention
        Multi-Head
      Positional Encoding
      Transformer 구현
        Encoder
        Decoder
      GPT vs BERT
        생성 vs 이해
    Phase 5<br/>현대 LLM
      Scaling Laws
        Emergent Abilities
      사전학습
        Next Token Prediction
      RLHF
        SFT
        PPO
      In-Context Learning
        Few-shot
        Chain-of-Thought
      최신 기법
        RAG
        Constitutional AI
</div>

<div style="margin: 32px 0; padding: 24px; background-color: #fff5e1; border-radius: 8px; border-left: 4px solid #fb8500;">
  <strong>📌 학습 전략:</strong> 각 Phase는 이전 Phase의 개념을 기반으로 쌓아 올립니다. 기초를 탄탄히 다진 후 다음 단계로 넘어가세요.
</div>

---

## 핵심 질문

각 Phase를 마치면 답할 수 있어야 하는 질문:

| Phase | 주차 | 이 단계를 마치면 답할 수 있어야 하는 질문 |
|:-----:|:---:|:----------------------------------------|
| 1 | 1-4주 | "컴퓨터가 어떻게 숫자들의 패턴을 스스로 찾아내지?" |
| 2 | 5-9주 | "신경망이 '학습'한다는 건 정확히 무슨 뜻이지?" |
| 3 | 10-14주 | "컴퓨터가 어떻게 '문맥'을 파악하지?" |
| 4 | 15-20주 | "왜 층을 깊이 쌓으면 더 복잡한 패턴을 잡아낼까?" |
| 5 | 21-26주 | "왜 같은 모델인데 프롬프트에 따라 결과가 달라지지?" |

---

## 학습 팁

| 카테고리 | 팁 |
|:--------|:---|
| 수학 걱정 줄이기 | 미분 = "기울기 = 얼마나 바꿔야 하는지 방향", 행렬 곱셈 = "여러 연산 한번에 묶기" |
| 안드로이드 개발자 강점 | 디버깅 능력 → 모델 학습도 "왜 안 되지?" 찾기, 아키텍처 감각 → 레이어 구조 이해 |
| 시간 배분 | 평일 출퇴근 2시간: 영상/논문, 주말 3-4시간: 코드 실습 |
| 확장 가능성 | on-device ML (TensorFlow Lite) 프로젝트로 안드로이드와 연결 |

---

자세한 리소스와 실습 내용은 [llm-learning-roadmap.md]({{ '/llm-learning-roadmap.md' | relative_url }})를 참고하세요.
