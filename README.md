# PleaseGiveMeAssets RAG Project
- **공통 사항**
  - 배포전 프로젝트 자동정렬 및 import 필수!!!(PyCharm CE)
    - 정렬
      - macOS : option + command + L
      - Window : Ctrl + Alt + L
    - import
      - macOS : control + option + O
      - Window : Ctrl + Alt + O
  - Git 배포시 commit message는 필수로 입력한다.
  - Git branch naming rule
    - feature : 기능 개발 브랜치 ex) feature/home
    - release : 버전 개발 브랜치 ex) release-1.0.0
    - hotfix : 긴급 개발 브랜치 ex) hotfix-1.0.0
    - 일반적으로 운영중이지 않은 소규모 프로젝트에서 hotfix는 사용할 필요가 없습니다. 대부분 feature 형태의 브랜치를 만들어 dev에 merge 해주시면 됩니다. release 브랜치는 테스트가 완료되면 팀장이 취합해 한번에 배포합니다.
    - git checkout -b feature/example origin/dev : dev를 기준으로 branch를 생성
  - 문의 사항은 Git issue나 comment, 디스코드를 통해 질문한다.
  - README.me 파일은 생각나는 기타 사항들을 모두 작성해주면 됩니다.

- **RAG 개발 가이드**
  - 링크 : https://bjhbhm.atlassian.net/wiki/spaces/cqdnDRC50KdZ/pages/4194305/RAG

#### 커밋 메시지 규칙 
|type | 사용예|
|-------|--------|
feat    | 새로운 기능 추가
docs	| 문서 수정
fix	    | 버그 수정
test    | 테스트 코드, 리팩토링 테스트 코드 추가
refactor| 코드 리팩토링
build   | 빌드 파일 수정
chore   | 빌드 업무 수정, 패키지 매니저 수정 (gitignore, ci/cd 관련 yaml파일 수정 등)
rename  | 파일 혹은 폴더명을 수정만 한 경우
remove  | 파일을 삭제만 한 경우

#### 예시
- {type}: 커멧메시지내용    
- 커밋 메시지 예시 - refactor: 로그인 기능 수정
