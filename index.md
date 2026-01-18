---
layout: home
title: ""
---

<div style="margin: 32px 0; padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; border-left: 5px solid #5a67d8;">
  <div id="quote-container" style="color: #fff;">
    <p id="quote-text" style="font-size: 18px; font-style: italic; margin: 0; line-height: 1.6;">
      "The journey of a thousand miles begins with a single step."
    </p>
    <p id="quote-korean" style="color: #e0e7ff; font-size: 16px; margin: 12px 0 0 0; line-height: 1.5;">
      "천리길도 한 걸음부터"
    </p>
    <p id="quote-author" style="color: #c7d2fe; font-size: 14px; margin: 12px 0 0 0; font-weight: 500;">
      — Lao Tzu (노자)
    </p>
  </div>    
</div>

<script>
// 학습과 성장에 관한 명언 모음 (한국어 번역 포함)
const quotes = [
  { text: "The journey of a thousand miles begins with a single step.", korean: "천리길도 한 걸음부터", author: "Lao Tzu (노자)" },
  { text: "Live as if you were to die tomorrow. Learn as if you were to live forever.", korean: "내일 죽을 것처럼 살고, 영원히 살 것처럼 배워라", author: "Mahatma Gandhi (마하트마 간디)" },
  { text: "The only way to do great work is to love what you do.", korean: "위대한 일을 하는 유일한 방법은 당신이 하는 일을 사랑하는 것이다", author: "Steve Jobs (스티브 잡스)" },
  { text: "In learning you will teach, and in teaching you will learn.", korean: "배우면서 가르치고, 가르치면서 배운다", author: "Phil Collins (필 콜린스)" },
  { text: "The expert in anything was once a beginner.", korean: "어떤 분야의 전문가도 한때는 초보자였다", author: "Helen Hayes (헬렌 헤이즈)" },
  { text: "Education is the most powerful weapon which you can use to change the world.", korean: "교육은 세상을 바꿀 수 있는 가장 강력한 무기이다", author: "Nelson Mandela (넬슨 만델라)" },
  { text: "The beautiful thing about learning is that no one can take it away from you.", korean: "배움의 아름다운 점은 아무도 그것을 빼앗아갈 수 없다는 것이다", author: "B.B. King (비비 킹)" },
  { text: "It does not matter how slowly you go as long as you do not stop.", korean: "멈추지만 않는다면 얼마나 천천히 가는지는 중요하지 않다", author: "Confucius (공자)" },
  { text: "Success is not final, failure is not fatal: it is the courage to continue that counts.", korean: "성공은 최종적인 것이 아니고, 실패는 치명적인 것이 아니다. 중요한 것은 계속할 용기다", author: "Winston Churchill (윈스턴 처칠)" },
  { text: "The only impossible journey is the one you never begin.", korean: "불가능한 여정은 시작하지 않는 여정뿐이다", author: "Tony Robbins (토니 로빈스)" },
  { text: "Tell me and I forget. Teach me and I remember. Involve me and I learn.", korean: "말해주면 잊고, 가르쳐주면 기억하고, 참여시키면 배운다", author: "Benjamin Franklin (벤자민 프랭클린)" },
  { text: "The capacity to learn is a gift; the ability to learn is a skill; the willingness to learn is a choice.", korean: "배울 수 있는 능력은 선물이고, 배우는 능력은 기술이며, 배우려는 의지는 선택이다", author: "Brian Herbert (브라이언 허버트)" },
  { text: "Intellectual growth should commence at birth and cease only at death.", korean: "지적 성장은 태어날 때 시작되어 죽을 때만 멈춰야 한다", author: "Albert Einstein (알베르트 아인슈타인)" },
  { text: "Anyone who stops learning is old, whether at twenty or eighty.", korean: "배움을 멈추는 사람은 스무 살이든 여든 살이든 늙은 것이다", author: "Henry Ford (헨리 포드)" },
  { text: "The more that you read, the more things you will know. The more that you learn, the more places you'll go.", korean: "더 많이 읽을수록 더 많이 알게 되고, 더 많이 배울수록 더 많은 곳에 갈 수 있다", author: "Dr. Seuss (닥터 수스)" },
  { text: "Learning never exhausts the mind.", korean: "배움은 결코 마음을 지치게 하지 않는다", author: "Leonardo da Vinci (레오나르도 다 빈치)" },
  { text: "Change is the end result of all true learning.", korean: "변화는 모든 진정한 배움의 최종 결과이다", author: "Leo Buscaglia (레오 버스카글리아)" },
  { text: "The mind is not a vessel to be filled, but a fire to be kindled.", korean: "마음은 채워야 할 그릇이 아니라 지펴야 할 불이다", author: "Plutarch (플루타르크)" },
  { text: "I am always doing that which I cannot do, in order that I may learn how to do it.", korean: "나는 항상 할 수 없는 것을 하는데, 그래야 하는 법을 배울 수 있기 때문이다", author: "Pablo Picasso (파블로 피카소)" },
  { text: "Wisdom is not a product of schooling but of the lifelong attempt to acquire it.", korean: "지혜는 학교 교육의 산물이 아니라 평생에 걸쳐 얻으려는 시도의 산물이다", author: "Albert Einstein (알베르트 아인슈타인)" },
  { text: "An investment in knowledge pays the best interest.", korean: "지식에 대한 투자가 가장 좋은 이자를 낸다", author: "Benjamin Franklin (벤자민 프랭클린)" },
  { text: "The roots of education are bitter, but the fruit is sweet.", korean: "교육의 뿌리는 쓰지만, 그 열매는 달다", author: "Aristotle (아리스토텔레스)" },
  { text: "Study the past if you would define the future.", korean: "미래를 정의하고 싶다면 과거를 공부하라", author: "Confucius (공자)" },
  { text: "I have no special talent. I am only passionately curious.", korean: "나에게는 특별한 재능이 없다. 나는 단지 열정적으로 호기심이 많을 뿐이다", author: "Albert Einstein (알베르트 아인슈타인)" },
  { text: "The best time to plant a tree was 20 years ago. The second best time is now.", korean: "나무를 심기에 가장 좋은 때는 20년 전이었다. 두 번째로 좋은 때는 바로 지금이다", author: "Chinese Proverb (중국 속담)" },
  { text: "What we learn with pleasure we never forget.", korean: "즐거움과 함께 배운 것은 결코 잊지 않는다", author: "Alfred Mercier (알프레드 머시어)" },
  { text: "I never lose. I either win or learn.", korean: "나는 절대 지지 않는다. 이기거나 배운다", author: "Nelson Mandela (넬슨 만델라)" },
  { text: "The future belongs to those who believe in the beauty of their dreams.", korean: "미래는 자신의 꿈이 아름답다고 믿는 사람들의 것이다", author: "Eleanor Roosevelt (엘리너 루스벨트)" },
  { text: "Don't watch the clock; do what it does. Keep going.", korean: "시계를 보지 말고 시계가 하는 일을 하라. 계속 가라", author: "Sam Levenson (샘 레벤슨)" },
  { text: "Everything you've ever wanted is on the other side of fear.", korean: "당신이 원했던 모든 것은 두려움 너머에 있다", author: "George Addair (조지 애데어)" }
];

// 날짜 기반으로 고정된 명언 선택 (매일 바뀌지만 하루 종일 동일)
const today = new Date();
const dayOfYear = Math.floor((today - new Date(today.getFullYear(), 0, 0)) / 1000 / 60 / 60 / 24);
const quoteIndex = dayOfYear % quotes.length;
const todayQuote = quotes[quoteIndex];

document.getElementById('quote-text').textContent = `"${todayQuote.text}"`;
document.getElementById('quote-korean').textContent = `"${todayQuote.korean}"`;
document.getElementById('quote-author').textContent = `— ${todayQuote.author}`;
</script>
