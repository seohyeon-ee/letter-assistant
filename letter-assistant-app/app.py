# app.py
import os
import json
from datetime import datetime
from typing import Dict, Optional

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

TONES = ["친근", "유머", "다정", "진지", "격식"]
LENGTHS = ["1~2문장", "5~6문장", "10문장 이상"]

TONE_GUIDE = {
    "친근": [
        "말투: 편하게, 과하게 가볍지 않게. 구어체 가능.",
        "문장 길이: 짧고 리듬감 있게.",
        "표현: 공감/맞장구(예: '나도 그 생각 했어')는 1~2번까지만.",
        "금지: 과한 감탄/오글거림, 과도한 이모지.",
    ],
    "유머": [
        "말투: 가볍게 웃길 수 있지만 상대를 놀리거나 비꼬지 말 것.",
        "유머 강도: 1~2번만 툭 치고, 본문 핵심은 진지하게 전달.",
        "기법: 과장/비유 1회 정도는 OK, 내부자 농담은 컨텍스트 없으면 금지.",
        "금지: 조롱, 비하, 공격적 농담, 민감 주제(외모/정치/혐오 등).",
    ],
    "다정": [
        "말투: 따뜻하고 배려 있게. 상대 감정을 먼저 인정/공감.",
        "감정 단어: 문단당 1~2개 정도로 과하지 않게.",
        "표현: '고마워/소중해/응원해' 같은 직접 표현 1~2회는 선명하게.",
        "금지: 장황한 감정 과잉, 뻔한 미사여구 남발.",
    ],
    "진지": [
        "말투: 차분하고 또렷하게. 핵심 메시지를 앞쪽에 명확히.",
        "구성: 이유/맥락 → 결론/요청 순으로 논리적으로.",
        "문장: 군더더기 없이 단정하게.",
        "금지: 가벼운 농담, 지나친 완곡 표현으로 핵심 흐리기.",
    ],
    "격식": [
        "말투: 존댓말, 공손하고 단정하게. 호칭/호격 일관성 유지.",
        "표현: 요청/제안은 완곡하게(예: '가능하실지 여쭙습니다').",
        "구성: 인사/배경 → 핵심 → 마무리 인사.",
        "금지: 반말/구어체/이모지/과한 감정 표현.",
    ],
}

TONE_STRENGTH = {
    "친근": "톤 강도: 중간 이상(친근함이 충분히 느껴지게).",
    "유머": "톤 강도: 강하게(유머 포인트는 1~2번 확실히).",
    "다정": "톤 강도: 강하게(다정함이 분명히 느껴지게).",
    "진지": "톤 강도: 강하게(진중하고 단정한 분위기 유지).",
    "격식": "톤 강도: 매우 강하게(공손/격식 유지, 흐트러짐 금지).",
}

TONE_TEMPERATURE = {
    "친근": 0.7,
    "유머": 0.9,
    "다정": 0.7,
    "진지": 0.5,
    "격식": 0.4,
}


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_business_relation(rel: str) -> bool:
    return rel in ["고객", "상사", "동료"]


def reset_all():
    st.session_state.profile = {
        "relation": "친구",
        "salutation": "",
        "purpose": "감사",
        "tone": "친근",
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
            "tone": "친근",
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
    blocks = [
        parts.get("intro", "").strip(),
        parts.get("body", "").strip(),
        parts.get("closing", "").strip(),
    ]
    blocks = [b for b in blocks if b]
    return "\n\n".join(blocks).strip()


def split_draft_to_parts(text: str) -> Dict[str, str]:
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
    if not api_key:
        return "⚠️ 사이드바에 OpenAI API Key를 입력해 주세요."
    if OpenAI is None:
        return "⚠️ openai 패키지가 설치되어 있지 않습니다. `pip install openai` 후 다시 실행해 주세요."

    client = OpenAI(api_key=api_key)
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

    tone = p["tone"]
    tone_rules = "\n".join([f"- {r}" for r in TONE_GUIDE.get(tone, [])])
    tone_strength = TONE_STRENGTH.get(tone, "")

    system = f"""
너는 한국어 편지 작성 어시스턴트다.
사용자의 입력(관계/목적/톤/분량/핵심 메시지/사실)을 바탕으로 자연스럽고 사람 같은 편지를 작성한다.

[톤 적용 규칙]
- 선택된 톤: {tone}
- {tone_strength}
{tone_rules}
- 톤은 문장 곳곳에 '지속적으로' 반영하되, 부자연스럽게 과장하지 말 것.

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

    tone = st.session_state.profile["tone"]
    temp = TONE_TEMPERATURE.get(tone, 0.7)

    draft = call_gpt(
        system=prompts["system"],
        user=prompts["user"],
        api_key=api_key,
        model="gpt-4.1-mini",
        temperature=temp,
    )

    if st.session_state.settings["polish_on"]:
        draft = polish_draft(draft)

    return draft.strip()


def polish_draft(draft: str) -> str:
    api_key = st.session_state.settings["api_key"]
    tone = st.session_state.profile["tone"]

    tone_rules = "\n".join([f"- {r}" for r in TONE_GUIDE.get(tone, [])])
    tone_strength = TONE_STRENGTH.get(tone, "")

    system = f"""
너는 한국어 편지 편집자다. 사용자가 쓴 듯 자연스럽게 다듬어라.
- 의미는 유지하고, 어색한 문장/중복/늘어짐을 고친다.
- 말투/호칭 일관성 유지.
- '톤'이 약해지지 않게 유지/강화한다.

[톤 유지 규칙]
- 선택된 톤: {tone}
- {tone_strength}
{tone_rules}

- 결과는 편지 본문만 출력.
""".strip()

    user = f"""
아래 편지를 더 자연스럽게 다듬어줘. 톤은 유지하거나 더 선명하게 만들어줘.

[편지]
{draft}
""".strip()

    return call_gpt(system=system, user=user, api_key=api_key, model="gpt-4.1-mini", temperature=0.3).strip()


def rewrite_with_new_tone(new_tone: str) -> str:
    api_key = st.session_state.settings["api_key"]
    base = st.session_state.draft.strip()

    tone_rules = "\n".join([f"- {r}" for r in TONE_GUIDE.get(new_tone, [])])
    tone_strength = TONE_STRENGTH.get(new_tone, "")

    system = f"""
너는 한국어 편지 작성 어시스턴트다.
주어진 편지를 '내용은 유지'하되, 요청한 톤으로 자연스럽게 재작성하라.

[톤 적용 규칙]
- 선택된 톤: {new_tone}
- {tone_strength}
{tone_rules}
- 톤은 문장 곳곳에 지속적으로 반영.

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

    temp = TONE_TEMPERATURE.get(new_tone, 0.6)
    out = call_gpt(system=system, user=user, api_key=api_key, model="gpt-4.1-mini", temperature=temp)
    if st.session_state.settings["polish_on"]:
        out = polish_draft(out)
    return out.strip()


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
        help="OPENAI API Key를 입력하세요. (배포 시엔 Secrets 사용 권장)",
    )
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
        if st.sidebar.button("불러오기", use_container_width=True, disabled=(picked == 0)):
            v = versions[picked - 1]
            st.session_state.profile = v["profile"]
            st.session_state.inputs = v["inputs"]
            st.session_state.draft = v["draft"]
            st.session_state.draft_parts = split_draft_to_parts(v["draft"])
            st.session_state["__draft_edit"] = loaded
            st.session_state["__final_text"] = loaded
            st.rerun()

    with col_b:
        if st.sidebar.button("새 편지 시작", use_container_width=True):
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
# UI: Single page (scroll)
# =============================
def header():
    st.title("6 letters")
    st.caption("한 화면에서 위→아래로 스크롤하며 입력하고, 초안을 생성한 뒤 재작성/내보내기까지 할 수 있어요.")


def render_basic_info():
    st.subheader("기본 정보")
    p = st.session_state.profile

    relation = st.selectbox(
        "관계",
        RELATIONS,
        index=RELATIONS.index(p["relation"]) if p["relation"] in RELATIONS else 0,
        key="__relation",
    )

    # 비즈니스 관계면 톤 프리셋 조정
    if relation != p["relation"]:
        p["relation"] = relation
        if is_business_relation(relation) and p["tone"] in ["친근", "유머", "다정"]:
            p["tone"] = "격식"

    salutation = st.text_input("호칭(예: 민수야 / OOO님)", value=p.get("salutation", ""), key="__salutation")

    purpose = st.radio(
        "편지 목적",
        PURPOSES,
        index=PURPOSES.index(p["purpose"]) if p["purpose"] in PURPOSES else 0,
        horizontal=True,
        key="__purpose",
    )

    tone = st.radio(
        "톤",
        TONES,
        index=TONES.index(p["tone"]) if p["tone"] in TONES else 0,
        horizontal=True,
        key="__tone",
    )

    length = st.selectbox(
        "분량",
        LENGTHS,
        index=LENGTHS.index(p["length"]) if p["length"] in LENGTHS else 1,
        key="__length",
    )

    p["relation"] = relation
    p["salutation"] = salutation
    p["purpose"] = purpose
    p["tone"] = tone
    p["length"] = length

    err = require_fields_ok()
    if err:
        st.warning(err)


def render_core_inputs():
    st.subheader("핵심 내용")
    i = st.session_state.inputs

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

    st.text_input(
        "피하고 싶은 내용(선택, 쉼표로 구분)",
        value=i.get("avoid", ""),
        key="__avoid",
        placeholder="예: 과한 감정표현, 장문, 이모지",
    )
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
    if err:
        st.warning(err)


def render_generate_actions():
    st.subheader("초안 생성")
    err1 = require_fields_ok()
    err2 = require_core_ok()
    disabled = bool(err1 or err2)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("초안 생성", type="primary", use_container_width=True, disabled=disabled):
            with st.spinner("초안을 생성 중..."):
                draft = generate_draft()
            st.session_state.draft = draft
            st.session_state.draft_parts = split_draft_to_parts(draft)
            st.session_state["__draft_edit"] = draft
            st.session_state["__final_text"] = draft
            st.rerun()

    with col2:
        if st.button("입력 초기화(새 편지)", use_container_width=True):
            reset_all()
            st.rerun()

    if err1:
        st.info(f"먼저 기본 정보를 완성해 주세요: {err1}")
    if err2:
        st.info(f"먼저 핵심 내용을 완성해 주세요: {err2}")


def render_draft_and_edit():
    st.subheader("초안 / 재작성")
    if not st.session_state.draft.strip():
        st.info("아직 초안이 없어요. 위에서 ‘초안 생성’을 눌러 주세요.")
        return

    p = st.session_state.profile

    st.markdown("**작업**")
    col1, col2 = st.columns([1, 2])

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
        
            st.session_state["__draft_edit"] = out
            st.session_state["__final_text"] = out
            
            st.rerun()

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


def render_export_and_versions():
    st.subheader("최종 / 내보내기")
    if not st.session_state.draft.strip():
        st.info("초안이 있어야 내보내기를 할 수 있어요.")
        return

    final_text = st.text_area("최종 편집", value=st.session_state.draft, height=320, key="__final_text")
    st.session_state.draft = final_text
    st.session_state.draft_parts = split_draft_to_parts(final_text)

    st.markdown("**복사하기**")
    st.code(final_text, language=None)

    st.download_button(
        "TXT 다운로드",
        data=final_text.encode("utf-8"),
        file_name=f"letter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

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


# =============================
# Main
# =============================
def main():
    st.set_page_config(page_title="6 letters", page_icon="✉️", layout="wide")
    init_state()
    render_sidebar()
    header()

    # 스크롤형 단일 페이지 구성
    with st.container():
        render_basic_info()
        st.divider()

        render_core_inputs()
        st.divider()

        render_generate_actions()
        st.divider()

        render_draft_and_edit()
        st.divider()

        render_export_and_versions()


if __name__ == "__main__":
    main()



