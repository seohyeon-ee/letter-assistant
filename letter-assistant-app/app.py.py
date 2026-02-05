# app.py
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st

# -----------------------------
# Optional: OpenAI SDK
# -----------------------------
# pip install openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =============================
# Helpers
# =============================
RELATIONS = ["친구", "연인", "부모", "선생님", "동료", "상사", "고객", "기타"]
PURPOSES = ["감사", "사과", "응원", "축하", "요청", "근황", "이별", "기타"]
TONES = ["담백", "다정", "진지", "캐주얼", "격식"]
LENGTHS = ["1~2문장", "5~6문장", "10문장 이상"]
EDIT_TARGETS = ["도입", "핵심 메시지 문단", "마무리"]


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_business_relation(rel: str) -> bool:
    return rel in ["고객", "상사", "동료"]


def reset_all():
    st.session_state.step = 1
    st.session_state.profile = {
        "relation": "친구",
        "salutation": "",
        "purpose": "감사",
        "tone": "담백",
        "length": "5~6문장",
    }
    st.session_state.inputs = {
        "core_message": "",
        "facts": ["", "", ""],
        "avoid": "",
        "context": "",
    }
    st.session_state.draft = ""
    st.session_state.draft_parts = {"intro": "", "body": "", "closing": ""}
    st.session_state.versions = []


def init_state():
    if "step" not in st.session_state:
        st.session_state.step = 1

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "polish_on": True,
            "reduce_cliche": True,
        }

    if "profile" not in st.session_state:
        st.session_state.profile = {
            "relation": "친구",
            "salutation": "",
            "purpose": "감사",
            "tone": "담백",
            "length": "5~6문장",
        }

    if "inputs" not in st.session_state:
        st.session_state.inputs = {
            "core_message": "",
            "facts": ["", "", ""],
            "avoid": "",
            "context": "",
        }

    if "draft" not in st.session_state:
        st.session_state.draft = ""

    if "draft_parts" not in st.session_state:
        st.session_state.draft_parts = {"intro": "", "body": "", "closing": ""}

    if "versions" not in st.session_state:
        st.session_state.versions = []


def join_draft(parts: Dict[str, str]) -> str:
    # parts: intro, body, closing
    blocks = [parts.get("intro", "").strip(), parts.get("body", "").strip(), parts.get("closing", "").strip()]
    blocks = [b for b in blocks if b]
    return "\n\n".join(blocks).strip()


def split_draft_to_parts(text: str) -> Dict[str, str]:
    """
    단순 분리: 빈 줄 기준으로 문단을 나눠,
    1문단=도입, 중간=본문(합침), 마지막=마무리로 매핑.
    (LLM이 JSON으로 parts를 주면 더 좋지만, 프로젝트용으로는 이 정도도 충분)
    """
    paras = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    if not paras:
        return {"intro": "", "body": "", "closing": ""}

    if len(paras) == 1:
        return {"intro": paras[0], "body": "", "closing": ""}

    if len(paras) == 2:
        return {"intro": paras[0], "body": paras[1], "closing": ""}

    intro = paras[0]
    closing = paras[-1]
    body = "\n\n".join(paras[1:-1]).strip()
    return {"intro": intro, "body": body, "closing": closing}


def require_fields_ok() -> Optional[str]:
    p = st.session_state.profile
    if not p.get("relation"):
        return "관계를 선택해 주세요."
    if not p.get("salutation", "").strip():
        return "호칭(예: 민수야 / OOO님)을 입력해 주세요."
    if not p.get("purpose"):
        return "편지 목적을 선택해 주세요."
    return None


def require_core_ok() -> Optional[str]:
    msg = st.session_state.inputs.get("core_message", "").strip()
    if not msg:
        return "핵심 메시지를 입력해 주세요."
    return None


# =============================
# GPT Call
# =============================
def call_gpt(system: str, user: str, api_key: str, model: str = "gpt-4.1-mini", temperature: float = 0.7) -> str:
    """
    - OpenAI Python SDK(최신) 기준
    - 환경에 SDK가 없으면 안내 메시지 반환
    """
    if not api_key:
        return "⚠️ 사이드바에 OpenAI API Key를 입력해 주세요."
    if OpenAI is None:
        return "⚠️ openai 패키지가 설치되어 있지 않습니다. `pip install openai` 후 다시 실행해 주세요."

    client = OpenAI(api_key=api_key)

    # Chat Completions 호환 형태 (Responses API로 바꿔도 됨)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def build_prompt_common() -> Dict[str, str]:
    s = st.session_state.settings
    p = st.session_state.profile
    i = st.session_state.inputs

    facts = [f.strip() for f in i["facts"] if f.strip()]
    avoid = [a.strip() for a in i["avoid"].split(",") if a.strip()]

    constraints = []
    if s["reduce_cliche"]:
        constraints.append("- 진부한 표현(항상/늘/진심으로/마음 깊이 등)을 과하게 쓰지 말 것.")
    constraints.append("- 사용자가 주지 않은 구체적 사실(날짜/장소/대화/사건)을 지어내지 말 것.")
    if avoid:
        constraints.append(f"- 다음 요소는 피할 것: {', '.join(avoid)}")

    length_guide = {
        "1~2문장": "아주 짧게 1~2문장.",
        "5~6문장": "적당히 5~6문장.",
        "10문장 이상": "상세하게 10문장 이상(하지만 장황하지 않게).",
    }.get(p["length"], "적당한 길이.")

    system = f"""
너는 한국어 편지 작성 어시스턴트다.
사용자의 입력(관계/목적/톤/분량/핵심 메시지/사실)을 바탕으로 자연스럽고 사람 같은 편지를 작성한다.
아래 규칙을 반드시 지켜라.
{chr(10).join(constraints)}
- 호칭과 말투는 끝까지 일관되게 유지할 것.
- 목적(감사/사과/요청 등)이 흐려지지 않게 중심을 잡을 것.
- 결과는 '편지 본문만' 출력(제목/해설/메타설명 금지).
""".strip()

    user = f"""
[편지 설정]
- 관계: {p['relation']}
- 호칭: {p['salutation']}
- 목적: {p['purpose']}
- 톤: {p['tone']}
- 분량: {p['length']} ({length_guide})

[핵심 메시지]
{i['core_message'].strip()}

[반드시 포함할 사실(있으면 활용)]
{json.dumps(facts, ensure_ascii=False)}

[추가 컨텍스트(있으면 참고)]
{i['context'].strip()}
""".strip()

    return {"system": system, "user": user}


def generate_draft() -> str:
    prompts = build_prompt_common()
    api_key = st.session_state.settings["api_key"]

    draft = call_gpt(
        system=prompts["system"],
        user=prompts["user"],
        api_key=api_key,
        # 모델은 상황에 맞게 바꿔도 됨
        model="gpt-4.1-mini",
        temperature=0.7,
    )

    # 옵션: 자동 검수/다듬기
    if st.session_state.settings["polish_on"]:
        draft = polish_draft(draft)

    return draft.strip()


def polish_draft(draft: str) -> str:
    api_key = st.session_state.settings["api_key"]
    system = """
너는 한국어 편지 편집자다. 사용자가 쓴 듯 자연스럽게 다듬어라.
- 의미는 유지하고, 어색한 문장/중복/늘어짐을 고친다.
- 말투/호칭 일관성 유지.
- 결과는 편지 본문만 출력.
""".strip()

    user = f"""
아래 편지를 더 자연스럽게 다듬어줘.

[편지]
{draft}
""".strip()

    return call_gpt(system=system, user=user, api_key=api_key, model="gpt-4.1-mini", temperature=0.4).strip()


def rewrite_with_new_tone(new_tone: str) -> str:
    # 기존 초안을 새 톤으로만 재작성
    api_key = st.session_state.settings["api_key"]
    p = st.session_state.profile.copy()
    p["tone"] = new_tone

    base = st.session_state.draft.strip()
    system = """
너는 한국어 편지 작성 어시스턴트다.
주어진 편지를 '내용은 유지'하되, 요청한 톤으로 자연스럽게 재작성하라.
- 구체적 사실 추가 금지
- 호칭/말투 일관성
- 결과는 편지 본문만
""".strip()

    user = f"""
[요청 톤]
{new_tone}

[원문 편지]
{base}
""".strip()

    out = call_gpt(system=system, user=user, api_key=api_key, model="gpt-4.1-mini", temperature=0.6)
    if st.session_state.settings["polish_on"]:
        out = polish_draft(out)
    return out.strip()


def edit_part(part_key: str, instruction: str) -> Dict[str, str]:
    """
    part_key: intro/body/closing
    instruction: 사용자 지시(부드럽게/단호하게 등)
    """
    api_key = st.session_state.settings["api_key"]
    parts = st.session_state.draft_parts.copy()
    target_text = parts.get(part_key, "").strip()

    if not target_text:
        return parts

    system = """
너는 한국어 편지 편집자다.
사용자의 지시에 따라 '지정된 부분'만 수정하라.
- 의미/사실 유지
- 톤/호칭 일관성
- 결과는 수정된 문단 텍스트만 출력
""".strip()

    user = f"""
[수정 지시]
{instruction}

[수정 대상 문단]
{target_text}
""".strip()

    revised = call_gpt(system=system, user=user, api_key=api_key, model="gpt-4.1-mini", temperature=0.6).strip()
    parts[part_key] = revised
    return parts


# =============================
# UI: Sidebar
# =============================
def render_sidebar():
    st.sidebar.header("설정")

    st.sidebar.text_input(
        "ChatGPT API Key",
        type="password",
        key="__api_key_input",
        value=st.session_state.settings.get("api_key", ""),
        help="OPENAI API Key를 입력하세요. (로컬에서만 사용 권장)",
    )
    # 반영
    st.session_state.settings["api_key"] = st.session_state.__api_key_input

    st.sidebar.toggle("자동 검수/다듬기 켜기", value=st.session_state.settings["polish_on"], key="__polish_toggle")
    st.session_state.settings["polish_on"] = st.session_state.__polish_toggle

    st.sidebar.toggle("클리셰 줄이기", value=st.session_state.settings["reduce_cliche"], key="__cliche_toggle")
    st.session_state.settings["reduce_cliche"] = st.session_state.__cliche_toggle

    st.sidebar.divider()
    st.sidebar.header("히스토리")

    versions = st.session_state.versions
    labels = ["(없음)"] + [f"v{idx+1} · {v['ts']}" for idx, v in enumerate(versions)]
    picked = st.sidebar.selectbox("생성 버전", options=list(range(len(labels))), format_func=lambda x: labels[x])

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("불러오기", use_container_width=True, disabled=(picked == 0)):
            v = versions[picked - 1]
            st.session_state.profile = v["profile"]
            st.session_state.inputs = v["inputs"]
            st.session_state.draft = v["draft"]
            st.session_state.draft_parts = split_draft_to_parts(v["draft"])
            st.session_state.step = 3
            st.rerun()

    with col_b:
        if st.button("새 편지 시작", use_container_width=True):
            reset_all()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.header("도움말")
    with st.sidebar.expander("좋은 입력 예시"):
        st.markdown(
            """
**핵심 메시지 예시**
- "요즘 많이 고마웠고, 다음엔 내가 더 챙기고 싶어."
- "지난번 일은 미안했고, 앞으로는 미리 얘기할게."
- "이번 제안은 A/B 두 옵션 중 B로 진행하고 싶습니다."

**반드시 포함할 사실 예시(최대 3개)**
- "지난주에 이사 도와줌"
- "이번 주 금요일 저녁 가능"
- "프로젝트 마감이 2/20"
"""
        )


# =============================
# UI: Steps
# =============================
def step_header():
    st.title("✉️ 편지 작성 어시스턴트")
    st.caption("단계별로 입력하고, 초안을 생성한 뒤 부분 수정/재작성으로 완성해요.")


def render_step_1():
    st.subheader("1) 기본 정보")

    p = st.session_state.profile

    relation = st.selectbox("관계", RELATIONS, index=RELATIONS.index(p["relation"]) if p["relation"] in RELATIONS else 0)
    # 비즈니스 관계면 톤 프리셋
    if relation != p["relation"]:
        p["relation"] = relation
        if is_business_relation(relation) and p["tone"] in ["담백", "다정", "캐주얼"]:
            p["tone"] = "격식"

    salutation = st.text_input("호칭(예: 민수야 / OOO님)", value=p.get("salutation", ""))
    purpose = st.radio("편지 목적", PURPOSES, index=PURPOSES.index(p["purpose"]) if p["purpose"] in PURPOSES else 0, horizontal=True)
    tone = st.radio("톤", TONES, index=TONES.index(p["tone"]) if p["tone"] in TONES else 0, horizontal=True)
    length = st.selectbox("분량", LENGTHS, index=LENGTHS.index(p["length"]) if p["length"] in LENGTHS else 1)

    # 저장
    p["relation"] = relation
    p["salutation"] = salutation
    p["purpose"] = purpose
    p["tone"] = tone
    p["length"] = length

    err = require_fields_ok()
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("다음", type="primary", use_container_width=True, disabled=bool(err)):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if err:
            st.warning(err)


def render_step_2():
    st.subheader("2) 핵심 내용")

    i = st.session_state.inputs
    p = st.session_state.profile

    st.text_area(
        "핵심 메시지(필수)",
        value=i.get("core_message", ""),
        key="__core_message",
        placeholder="전달하고 싶은 결론 1~2문장",
        height=120,
    )
    i["core_message"] = st.session_state.__core_message

    st.markdown("**반드시 포함할 사실(최대 3개)**")
    cols = st.columns(3)
    for idx in range(3):
        with cols[idx]:
            key = f"__fact_{idx}"
            st.text_input(f"사실 {idx+1}", value=i["facts"][idx], key=key)
            i["facts"][idx] = st.session_state[key]

    st.text_input("피하고 싶은 내용(선택, 쉼표로 구분)", value=i.get("avoid", ""), key="__avoid", placeholder="예: 과한 감정표현, 장문, 이모지")
    i["avoid"] = st.session_state.__avoid

    with st.expander("추가 컨텍스트(선택)"):
        st.text_area(
            "상대와의 최근 상황/거리감/민감한 부분",
            value=i.get("context", ""),
            key="__context",
            height=120,
        )
        i["context"] = st.session_state.__context

    err = require_core_ok()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("초안 생성", type="primary", use_container_width=True, disabled=bool(err)):
            with st.spinner("초안을 생성 중..."):
                draft = generate_draft()
            st.session_state.draft = draft
            st.session_state.draft_parts = split_draft_to_parts(draft)
            st.session_state.step = 3
            st.rerun()

    if err:
        st.warning(err)


def render_step_3():
    st.subheader("3) 생성 결과 (초안)")

    if not st.session_state.draft.strip():
        st.info("아직 초안이 없어요. 2단계에서 초안을 생성해 주세요.")
        if st.button("2단계로 이동"):
            st.session_state.step = 2
            st.rerun()
        return

    p = st.session_state.profile

    # 상단: 버튼 줄
    st.markdown("**작업**")
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        if st.button("전체 재생성", use_container_width=True):
            with st.spinner("전체를 다시 생성 중..."):
                draft = generate_draft()
            st.session_state.draft = draft
            st.session_state.draft_parts = split_draft_to_parts(draft)
            st.rerun()

    with col2:
        new_tone = st.selectbox("톤만 바꿔 재작성", TONES, index=TONES.index(p["tone"]), key="__new_tone")
        if st.button("톤 변경 적용", use_container_width=True):
            with st.spinner("톤을 바꿔 재작성 중..."):
                out = rewrite_with_new_tone(new_tone)
            st.session_state.profile["tone"] = new_tone
            st.session_state.draft = out
            st.session_state.draft_parts = split_draft_to_parts(out)
            st.rerun()

    with col3:
        if st.button("최종 단계로", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()

    # 본문 편집
    st.markdown("**초안(편집 가능)**")
    edited = st.text_area(
        "편지 본문",
        value=st.session_state.draft,
        height=320,
        key="__draft_edit",
        label_visibility="collapsed",
    )
    if edited != st.session_state.draft:
        st.session_state.draft = edited
        st.session_state.draft_parts = split_draft_to_parts(edited)

    st.divider()

    # 부분 수정
    st.markdown("**부분 수정**")
    target = st.selectbox("수정하고 싶은 부분", EDIT_TARGETS, key="__edit_target")
    instruction = st.text_input("수정 지시", key="__edit_instruction", placeholder="예: 좀 더 부드럽게 / 더 단호하게 / 더 짧게")

    # mapping
    target_map = {
        "도입": "intro",
        "핵심 메시지 문단": "body",
        "마무리": "closing",
    }
    part_key = target_map[target]

    if st.button("선택 부분만 수정", use_container_width=True, disabled=not instruction.strip()):
        with st.spinner("선택 부분을 수정 중..."):
            new_parts = edit_part(part_key, instruction.strip())
        st.session_state.draft_parts = new_parts
        st.session_state.draft = join_draft(new_parts)
        st.rerun()

    # 미리보기(문단)
    with st.expander("문단 분리 미리보기"):
        parts = st.session_state.draft_parts
        st.markdown("**도입**")
        st.write(parts.get("intro", ""))
        st.markdown("**본문**")
        st.write(parts.get("body", ""))
        st.markdown("**마무리**")
        st.write(parts.get("closing", ""))

    # 네비게이션
    c1, c2 = st.columns(2)
    with c1:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with c2:
        if st.button("최종 단계로 이동", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def render_step_4():
    st.subheader("4) 최종 편집 / 내보내기")

    if not st.session_state.draft.strip():
        st.info("아직 초안이 없어요. 2단계에서 생성해 주세요.")
        if st.button("2단계로 이동"):
            st.session_state.step = 2
            st.rerun()
        return

    # 최종 편집
    final_text = st.text_area("최종 편집", value=st.session_state.draft, height=320, key="__final_text")
    st.session_state.draft = final_text
    st.session_state.draft_parts = split_draft_to_parts(final_text)

    st.markdown("**복사하기**")
    st.code(final_text, language=None)

    # 다운로드
    st.download_button(
        "TXT 다운로드",
        data=final_text.encode("utf-8"),
        file_name=f"letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # 버전 저장
    if st.button("버전 저장", type="primary", use_container_width=True):
        st.session_state.versions.append(
            {
                "profile": st.session_state.profile.copy(),
                "inputs": {
                    "core_message": st.session_state.inputs["core_message"],
                    "facts": st.session_state.inputs["facts"][:],
                    "avoid": st.session_state.inputs["avoid"],
                    "context": st.session_state.inputs["context"],
                },
                "draft": st.session_state.draft,
                "ts": now_ts(),
            }
        )
        st.success(f"저장 완료! (v{len(st.session_state.versions)})")

    # 네비게이션
    c1, c2 = st.columns(2)
    with c1:
        if st.button("이전", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with c2:
        if st.button("처음으로", use_container_width=True):
            st.session_state.step = 1
            st.rerun()


# =============================
# Main
# =============================
def main():
    st.set_page_config(page_title="편지 작성 어시스턴트", page_icon="✉️", layout="wide")
    init_state()
    render_sidebar()
    step_header()

    step = st.session_state.step
    if step == 1:
        render_step_1()
    elif step == 2:
        render_step_2()
    elif step == 3:
        render_step_3()
    elif step == 4:
        render_step_4()
    else:
        st.session_state.step = 1
        st.rerun()


if __name__ == "__main__":
    main()
