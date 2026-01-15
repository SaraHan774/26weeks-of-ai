---
title: "Attention is All You Need 논문 읽기"
date: 2026-01-16
---

## 역사를 바꾼 논문

이번 주는 2017년 Google이 발표한 "Attention is All You Need" 논문을 읽고 이해하는 주입니다.

### 왜 중요한가?

이 논문은 현대 LLM의 기초가 되는 **Transformer** 아키텍처를 소개했습니다.
- GPT, BERT, Claude, ChatGPT 모두 Transformer 기반
- RNN/LSTM을 대체하는 새로운 패러다임

### 핵심 개념

#### 1. Self-Attention

문장 내 단어들이 서로 어떻게 관련되어 있는지 계산하는 메커니즘

```python
# Self-Attention의 핵심 아이디어 (의사 코드)
def self_attention(query, key, value):
    # 1. Query와 Key의 유사도 계산
    scores = matmul(query, key.T) / sqrt(d_k)

    # 2. Softmax로 확률 분포 변환
    weights = softmax(scores)

    # 3. Value에 가중치 적용
    output = matmul(weights, value)

    return output
```

#### 2. Multi-Head Attention

여러 개의 Attention을 병렬로 수행하여 다양한 관점에서 정보 포착

#### 3. Positional Encoding

단어의 순서 정보를 임베딩에 추가
- RNN과 달리 Transformer는 순서 정보가 없음
- Sin/Cos 함수로 위치 정보 인코딩

### Jay Alammar의 "Illustrated Transformer"

논문과 함께 [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 블로그를 읽었습니다.
시각화가 정말 훌륭해서 개념 이해에 큰 도움이 되었습니다!

### 이해한 것

- ✅ Attention이 "어디에 집중할지" 결정하는 메커니즘임을 이해
- ✅ Query, Key, Value의 역할 파악
- ✅ 왜 RNN보다 병렬화에 유리한지 이해

### 아직 어려운 것

- ❓ Positional Encoding의 수식이 왜 Sin/Cos인지
- ❓ Multi-Head가 정확히 어떻게 다른 관점을 제공하는지
- ❓ Layer Normalization vs Batch Normalization

### 주말 실습 계획

논문의 Figure 1(Transformer 아키텍처)을 직접 그리면서 각 컴포넌트의 역할을 정리할 예정입니다.

### 참고 자료

- 📄 [Attention is All You Need](https://arxiv.org/abs/1706.03762) (원본 논문)
- 📝 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
- 🎬 [Transformer Neural Networks Explained](https://www.youtube.com/watch?v=TQQlZhbC5ps) (Computerphile)
