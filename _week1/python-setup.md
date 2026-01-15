---
title: "Python 환경 세팅하기"
date: 2026-01-16
---

## Python 개발 환경 구축

이번 주 첫 번째 과제는 Python 개발 환경을 세팅하는 것입니다.

### 설치할 것들

1. **Python 3.x**: [python.org](https://www.python.org)에서 최신 버전 다운로드
2. **VSCode**: 코드 에디터
3. **필수 패키지**: numpy, pandas, matplotlib

### 설치 명령어

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 기본 패키지 설치
pip install numpy pandas matplotlib jupyter
```

### 첫 번째 Python 프로그램

```python
# hello_world.py
print("Hello, LLM Learning Journey!")

# 기본 변수 선언
name = "Gahee"
weeks = 26

print(f"{name}님의 {weeks}주 학습 여정을 시작합니다!")
```

### 학습 노트

- Kotlin과 달리 Python은 동적 타입 언어
- 들여쓰기로 블록을 구분 (중괄호 사용 안 함)
- None은 Kotlin의 null과 유사

### 다음 단계

내일은 Python 기본 문법을 더 깊이 공부할 예정입니다.
