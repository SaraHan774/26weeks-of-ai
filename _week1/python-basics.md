---
title: "Python 기초 문법 익히기"
date: 2026-01-16
---

## Python vs Kotlin 비교

안드로이드 개발자 관점에서 Python과 Kotlin의 차이점을 정리합니다.

### 변수 선언

**Kotlin:**
```kotlin
val name: String = "Gahee"
var age: Int = 30
```

**Python:**
```python
name = "Gahee"  # 타입 선언 불필요
age = 30
```

### 리스트 다루기

**Kotlin:**
```kotlin
val numbers = listOf(1, 2, 3, 4, 5)
val doubled = numbers.map { it * 2 }
```

**Python:**
```python
numbers = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in numbers]  # 리스트 컴프리헨션
```

### 함수 정의

**Kotlin:**
```kotlin
fun greet(name: String): String {
    return "Hello, $name"
}
```

**Python:**
```python
def greet(name):
    return f"Hello, {name}"
```

### None 처리

**Kotlin:**
```kotlin
val name: String? = null
val length = name?.length ?: 0
```

**Python:**
```python
name = None
length = len(name) if name is not None else 0
```

## 핵심 학습 포인트

1. **동적 타입**: 실행 시점에 타입이 결정됨
2. **들여쓰기**: 코드 블록을 나타내는 유일한 방법 (4칸 스페이스 권장)
3. **리스트 컴프리헨션**: 간결한 리스트 생성 방법
4. **언패킹**: `a, b = [1, 2]` 같은 편리한 문법

## 오늘의 실습

점프 투 파이썬의 2-3장을 읽고 간단한 CLI 프로그램을 만들어봤습니다.

```python
# word_counter.py
def count_words(text):
    words = text.split()
    return len(words)

if __name__ == "__main__":
    sample = "Python is easy to learn"
    print(f"단어 개수: {count_words(sample)}")
```

## 내일 계획

NumPy 기초를 배우고 배열 연산에 익숙해질 예정입니다.
