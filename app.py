import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ----------------- UI 및 제목 설정 -----------------
st.set_page_config(page_title="과학실 물품 변동 감지 시스템", layout="wide")
st.title("🧪 과학실 물품 도난 및 위치 변동 감지 시스템")
st.markdown("비포(기준) 사진과 애프터(현재) 사진을 비교하여 위치가 바뀌거나 사라진 물건을 찾아냅니다.")
st.divider()

# ----------------- 사이드바 설정 (민감도 조절) -----------------
st.sidebar.header("⚙️ 감지 설정")
# 조명 변화 등에 유연하게 대처하기 위한 임계값 조절 슬라이더
min_area = st.sidebar.slider("감지할 최소 크기 (픽셀 면적)", min_value=100, max_value=5000, value=800, step=100)
threshold_val = st.sidebar.slider("명암 차이 민감도 (낮을수록 민감)", min_value=10, max_value=100, value=25, step=5)

# ----------------- 이미지 업로드 섹션 -----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 1. 기준 사진 (Before)")
    before_file = st.file_uploader("과학실의 원래 상태 사진을 업로드하세요.", type=["jpg", "jpeg", "png"], key="before")
    if before_file:
        st.image(before_file, caption="기준 상태", use_container_width=True)

with col2:
    st.subheader("🔍 2. 현재 사진 (After)")
    after_file = st.file_uploader("확인할 현재 상태 사진을 업로드하세요.", type=["jpg", "jpeg", "png"], key="after")
    if after_file:
        st.image(after_file, caption="현재 상태", use_container_width=True)

st.divider()

# ----------------- 변동 감지 및 결과 출력 -----------------
st.subheader("🚨 분석 결과")

if before_file and after_file:
    # PIL 이미지를 OpenCV 이미지(numpy array)로 변환
    img1 = Image.open(before_file)
    img2 = Image.open(after_file)
    
    # OpenCV에서 다루기 위해 numpy 배열로 변환 (RGB -> BGR)
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    
    # 크기 일치시키기
    img2_cv = cv2.resize(img2_cv, (img1_cv.shape[1], img1_cv.shape[0]))
    
    # 결과 표시용 복사본
    result_img = img2_cv.copy()

    # 이미지 전처리 (흑백화 + 블러)
    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # 두 이미지의 차이 계산
    frame_delta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    change_count = 0
    for contour in contours:
        # 설정한 최소 크기보다 작은 변화는 무시
        if cv2.contourArea(contour) < min_area:
            continue
            
        change_count += 1
        # 다른 부분에 빨간색 사각형 그리기
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (255, 0, 0), 4) # Streamlit은 RGB 기준이므로 빨간색은 (255,0,0)

    # 결과 전송 (BGR -> RGB 변환 후 Streamlit 표시)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    if change_count > 0:
        st.error(f"⚠️ 경고: 위치가 변경되거나 사라진 영역이 {change_count}군데 감지되었습니다!")
        st.image(result_img_rgb, caption="변동 위치가 빨간색 사각형으로 표시된 결과", use_container_width=True)
    else:
        st.success("✅ 안전합니다: 변경된 사항이 감지되지 않았습니다.")
        st.image(result_img_rgb, caption="변동 없음", use_container_width=True)

else:
    st.info("💡 왼쪽과 오른쪽 모두에 사진을 업로드하면 자동으로 분석이 시작됩니다.")
