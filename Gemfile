source "https://rubygems.org"

# Jekyll 버전
gem "jekyll", "~> 4.3"

# 기본 테마
gem "minima", "~> 2.5"

# Jekyll 플러그인
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-seo-tag", "~> 2.8"
end

# Windows와 JRuby에서 시간대 정보 필요
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Windows에서 디렉토리 감시
gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]

# JRuby에서 http_parser.rb 사용
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# GitHub Pages 호환 (선택사항)
# gem "github-pages", group: :jekyll_plugins
