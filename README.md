# 26주 LLM 학습 여정

비전공 안드로이드 개발자의 LLM 원리 이해를 위한 26주 학습 기록 블로그입니다.

## 프로젝트 구조

```
26weeks-of-ai/
├── _config.yml          # Jekyll 설정
├── Gemfile              # Ruby 의존성
├── index.md             # 메인 페이지
├── llm-learning-roadmap.md  # 전체 학습 로드맵
├── _layouts/            # 페이지 레이아웃
│   ├── default.html
│   ├── home.html
│   └── post.html
├── _includes/           # 재사용 컴포넌트
│   └── navigation.html
├── assets/              # CSS, 이미지 등
│   └── css/
│       └── style.css
└── _week1/ ~ _week26/   # 주차별 학습 내용
    ├── topic1.md
    ├── topic2.md
    └── ...
```

## 로컬 실행 방법

### 1. 의존성 설치

```bash
# Ruby와 Bundler가 설치되어 있어야 합니다
bundle install
```

### 2. Jekyll 서버 실행

```bash
bundle exec jekyll serve
```

브라우저에서 `http://localhost:4000` 접속

### 3. 빌드 (배포용)

```bash
bundle exec jekyll build
```

빌드된 파일은 `_site/` 폴더에 생성됩니다.

## 학습 내용 작성 방법

### 새 글 작성

각 주차 폴더(`_week1` ~ `_week26`)에 마크다운 파일을 생성합니다.

```markdown
---
title: "글 제목"
date: 2026-01-16
---

## 내용 작성

여기에 학습 내용을 작성합니다.
```

### Front Matter

각 글의 상단에 다음 정보를 포함해야 합니다:

- `title`: 글 제목 (필수)
- `date`: 작성 날짜 (선택)

`phase`와 `week`는 `_config.yml`에서 자동으로 설정됩니다.

## 학습 Phase

1. **Phase 1 (1-4주)**: Python과 기초 체력
2. **Phase 2 (5-9주)**: 딥러닝 기초
3. **Phase 3 (10-14주)**: 자연어 처리 기초
4. **Phase 4 (15-20주)**: Transformer 깊이 파기
5. **Phase 5 (21-26주)**: 현대 LLM의 비밀

자세한 로드맵은 `llm-learning-roadmap.md` 또는 [Roadmap 페이지](https://localhost:4000/roadmap)를 참고하세요.

학습에 필요한 모든 자료는 [Resources 페이지](https://localhost:4000/resources)에서 확인할 수 있습니다.

## GitHub Pages 배포

### 1. GitHub 저장소 생성

```bash
git remote add origin https://github.com/username/26weeks-of-ai.git
git branch -M main
git push -u origin main
```

### 2. GitHub Pages 활성화

1. 저장소 Settings > Pages
2. Source: Deploy from a branch
3. Branch: main, Folder: / (root)
4. Save

### 3. 사이트 URL

`https://username.github.io/26weeks-of-ai/`

## 기여

이 프로젝트는 개인 학습 기록이지만, 비슷한 여정을 시작하는 분들께 도움이 되길 바랍니다.

## 라이선스

MIT License
