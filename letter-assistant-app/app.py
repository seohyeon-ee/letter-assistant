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

# ✅ 최종 6개 톤
TONES = ["친근", "유머", "감성", "담백", "진지", "격식"]
LENGTHS = ["1~2문장", "5~6문장", "10문장 이상"]

# ✅ 톤별 결과물 박스 색상(채도 낮게)
TONE_COLORS = {
    "친근": "#FFF4CC",  # 연노랑
    "유머": "#DFF5E1",  # 연초록
    "감성": "#F8DDEA",  # 연분홍
    "담백": "#F1F3F5",  # 연회색
    "진지": "#DCEEFF",  # 연하늘
    "격식": "#E9E1FF",  # 연보라
}

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
    "감성": [
        "말투: 부드럽고 분위기 있게. 담담한 여운을 남길 것.",
        "표현: 이미지/비유는 1~2회까지, 과장은 금지.",
        "리듬: 문장을 너무 길게 늘이지 말고, 호흡이 느껴지게.",
        "금지: 과한 시적 허세, 진부한 미사여구 남발, 오글거림.",
    ],
    "담백": [
        "말투: 차분하고 깔끔하게. 꾸밈을 최소화.",
        "문장: 짧고 명확하게 핵심 위주.",
        "표현: 감정은 과하지 않게, 사실/의도는 또렷하게.",
        "금지: 과한 감탄, 장황함, 미사여구.",
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
    "감성": "톤 강도: 중간 이상(분위기/여운이 느껴지되 과장 금지).",
    "담백": "톤 강도: 중간(담담하고 깔끔하게, 과장 없이).",
    "진지": "톤 강도: 강하게(진중하고 단정한 분위기 유지).",
    "격식": "톤 강도: 매우 강하게(공손/격식 유지, 흐트러짐 금지).",
}

TONE_TEMPERATURE = {
    "친근": 0.7,
    "유머": 0.9,
    "감성": 0.8,
    "담백": 0.5,
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
    st.session_state.versions = []
    st.session_state["__draft_edit"] = ""
    st.session_state["__final_text"] = ""


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

    if "versions" not in st.session_state:
        st.session_state.versions = []

    if "__draft_edit" not in st.session_state:
        st.session_state["__draft_edit"] = st.session_state.draft
    if "__final_text" not in st.session_state:
        st.session_state["__final_text"] = st.session_state.draft


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


def apply_result_box_css(tone: str):
    """✅ '결과물' 박스만 톤 컬러 적용 (핵심내용/최종편집에는 미적용)"""
    bg = TONE_COLORS.get(tone, "#F1F3F5")
    st.markdown(
        f"""
<style>
/* 결과물 섹션만 색 적용 */
.result-box div[data-testid="stTextArea"] textarea {{
  background-color: {bg} !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


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
    return out.strip()


def set_draft(text: str):
    st.session_state.draft = text
    st.session_state["__draft_edit"] = text
    # 최종 편집은 별도 입력을 존중하고 싶으면 동기화하지 않아도 되지만,
    # 지금 UX는 "한 소스"로 유지하는 게 편해서 같이 맞춤.
    st.session_state["__final_text"] = text


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
            set_draft(v["draft"])
            st.rerun()

    with col_b:
        if st.sidebar.button("새 편지 시작", use_container_width=True):
            reset_all()
            st.rerun()


# =============================
# UI: Single page (scroll)
# =============================
def header():
    st.title("6 letters")
    st.caption("위→아래로 스크롤하며 입력하고, 초안을 생성한 뒤 톤 재작성/내보내기까지 할 수 있어요.")


def render_basic_info():
    st.subheader("기본 정보")
    p = st.session_state.profile

    relation = st.selectbox(
        "관계",
        RELATIONS,
        index=RELATIONS.index(p["relation"]) if p["relation"] in RELATIONS else 0,
        key="__relation",
    )

    if relation != p["relation"]:
        p["relation"] = relation
        if is_business_relation(relation) and p["tone"] in ["친근", "유머", "감성", "담백"]:
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
            set_draft(draft)
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
    st.subheader("결과물 / 톤 재작성")

    if not st.session_state.draft.strip():
        st.info("아직 결과물이 없어요. 위에서 ‘초안 생성’을 눌러 주세요.")
        return

    p = st.session_state.profile

    st.markdown("**작업**")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("전체 재생성", use_container_width=True):
            with st.spinner("전체를 다시 생성 중..."):
                draft = generate_draft()
            set_draft(draft)
            st.rerun()

    with col2:
        new_tone = st.selectbox("톤만 바꿔 재작성", TONES, index=TONES.index(p["tone"]), key="__new_tone")
        if st.button("톤 변경 적용", use_container_width=True):
            with st.spinner("톤을 바꿔 재작성 중..."):
                st.session_state.profile["tone"] = new_tone
                out = rewrite_with_new_tone(new_tone)
                if st.session_state.settings["polish_on"]:
                    out = polish_draft(out)
            set_draft(out)
            st.rerun()

    st.markdown("**결과물(편집 가능)**")

    # ✅ 결과물 박스만 톤 색상 적용: wrapper div + CSS
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    edited = st.text_area(
        "편지 본문",
        value=st.session_state.draft,
        height=320,
        key="__draft_edit",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if edited != st.session_state.draft:
        set_draft(edited)


def render_export_and_versions():
    st.subheader("최종 / 내보내기")
    if not st.session_state.draft.strip():
        st.info("결과물이 있어야 내보내기를 할 수 있어요.")
        return

    # ❌ 최종 편집은 색 적용 안 함 (일반 text_area 그대로)
    final_text = st.text_area("최종 편집", value=st.session_state.draft, height=320, key="__final_text")
    if final_text != st.session_state.draft:
        set_draft(final_text)

    st.markdown("**복사하기**")
    st.code(st.session_state.draft, language=None)

    st.download_button(
        "TXT 다운로드",
        data=st.session_state.draft.encode("utf-8"),
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

    # ✅ 결과물 박스에만 톤 컬러 적용
    apply_result_box_css(st.session_state.profile.get("tone", "담백"))

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
