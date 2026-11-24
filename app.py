import os
import tempfile
import io
import wave

import numpy as np
import pandas as pd
import parselmouth
import plotly.graph_objects as go
import streamlit as st


# 1. 웹사이트 기본 설정
st.set_page_config(
    page_title="Acoustic Prosody Analyzer",
    page_icon="🎵",
    layout="wide"
)

st.title("Acoustic Prosody Analyzer")
st.write("2026 영어음성학 4조 ㅣ 김예은, 최재현, 길민서, 김석준, 피이엔")
st.write("Upload a WAV file to compare F0, Intensity, and Duration.")


# 2. WAV 파일 업로드
uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)


# 3. 파일이 업로드되었을 때 실행
if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()

    st.subheader("Audio Player")
    st.audio(audio_bytes, format="audio/wav")

    temp_path = None

    try:
        # 4. 임시 파일 생성
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # 5. 음성 파일 읽기
        sound = parselmouth.Sound(temp_path)

        duration = sound.duration
        sample_rate = sound.sampling_frequency
        channels = sound.n_channels

        # 6. 기본 음성 정보 출력
        st.subheader("Audio Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Duration", f"{duration:.2f} s")
        col2.metric("Sample Rate", f"{sample_rate:.0f} Hz")
        col3.metric("Channels", channels)

        st.write("File name:", uploaded_file.name)

        # 7. 파형 데이터 준비
        samples = np.mean(sound.values, axis=0)
        times = np.arange(len(samples)) / sample_rate

        # 화면 출력용 데이터 축소
        step = max(1, len(samples) // 10000)

        df = pd.DataFrame({
            "Time (s)": times[::step],
            "Amplitude": samples[::step]
        })

        # 8. 파형 그래프 생성
        st.subheader("Audio Waveform")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Time (s)"],
                y=df["Amplitude"],
                mode="lines",
                name="Waveform"
            )
        )

        fig.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=400
        )

        # STEP 15. Waveform 영역 선택 기능
        st.plotly_chart(fig, width="stretch")        

        # 9. Fundamental Frequency (F0) 분석
        st.subheader("Fundamental Frequency (F0)")

        pitch = sound.to_pitch(
            time_step=0.01,
            pitch_floor=75,
            pitch_ceiling=600
        )

        # 시간 및 F0 값 가져오기
        pitch_times = pitch.xs()
        pitch_values = pitch.selected_array["frequency"]

        # 무성 구간의 0Hz를 NaN으로 변경
        pitch_values = np.where(
            pitch_values > 0,
            pitch_values,
            np.nan
        )

        # 유효한 F0 값만 추출
        valid_pitch = pitch_values[
            ~np.isnan(pitch_values)
        ]

        if len(valid_pitch) > 0:

            # F0 통계 계산
            mean_f0 = np.mean(valid_pitch)
            min_f0 = np.min(valid_pitch)
            max_f0 = np.max(valid_pitch)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Mean F0",
                f"{mean_f0:.2f} Hz"
            )
            col2.metric(
                "Minimum F0",
                f"{min_f0:.2f} Hz"
            )
            col3.metric(
                "Maximum F0",
                f"{max_f0:.2f} Hz"
            )

            # F0 그래프 생성
            fig_f0 = go.Figure()

            fig_f0.add_trace(
                go.Scatter(
                    x=pitch_times,
                    y=pitch_values,
                    mode="lines",
                    name="F0",
                    connectgaps=False
                )
            )

            fig_f0.update_layout(
                title="Pitch Contour",
                xaxis_title="Time (seconds)",
                yaxis_title="F0 (Hz)",
                height=400
            )

            st.plotly_chart(
                fig_f0,
                width="stretch"
            )

        else:
            st.warning(
                "유효한 F0를 검출하지 못했습니다."
            )
        
        # 10. Intensity 분석
        st.subheader("Intensity Analysis")

        # Parselmouth로 음향 강도 추출
        intensity = sound.to_intensity(
            minimum_pitch=100,
            time_step=0.01
        )

        # 시간 및 Intensity 값 가져오기
        intensity_times = intensity.xs()
        intensity_values = intensity.values[0]

        # 유효한 Intensity 값만 선택
        valid_intensity = intensity_values[
            np.isfinite(intensity_values)
        ]

        if len(valid_intensity) > 0:

            # Intensity 통계 계산
            mean_intensity = np.mean(valid_intensity)
            min_intensity = np.min(valid_intensity)
            max_intensity = np.max(valid_intensity)

            # 통계 정보 출력
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Mean Intensity",
                f"{mean_intensity:.2f} dB"
            )

            col2.metric(
                "Minimum Intensity",
                f"{min_intensity:.2f} dB"
            )

            col3.metric(
                "Maximum Intensity",
                f"{max_intensity:.2f} dB"
            )

            # Intensity 그래프 생성
            fig_intensity = go.Figure()

            fig_intensity.add_trace(
                go.Scatter(
                    x=intensity_times,
                    y=intensity_values,
                    mode="lines",
                    name="Intensity",
                    line=dict(color="orange")
                )
            )

            fig_intensity.update_layout(
                title="Intensity Contour",
                xaxis_title="Time (seconds)",
                yaxis_title="Intensity (dB)",
                height=400
            )

            st.plotly_chart(
                fig_intensity,
                width="stretch"
            )

        else:
            st.warning(
                "유효한 Intensity 값을 검출하지 못했습니다."
            )
        
        
        # =====================================
        # STEP 15. Manual Audio Region Selection
        # =====================================

        import hashlib

        st.subheader("Select Audio Region")

        st.write(
            "파형에서 구간을 드래그하거나 "
            "시작·종료 시간을 직접 입력하세요."
        )

        # 1. 파일마다 별도의 입력 상태 사용
        file_id = hashlib.md5(audio_bytes).hexdigest()[:12]

        start_key = f"region_start_{file_id}"
        end_key = f"region_end_{file_id}"
        selection_key = f"region_selection_{file_id}"
        chart_key = f"region_chart_{file_id}"

        # 처음 업로드한 파일의 기본 선택 범위
        if start_key not in st.session_state:
            st.session_state[start_key] = 0.0

        if end_key not in st.session_state:
            st.session_state[end_key] = float(duration)

        
        # =====================================
        # STEP 18. Select Audio Spectrogram
        # =====================================

        # 1. Praat / Parselmouth로 Spectrogram 생성
        max_frequency = min(
            5000.0,
            sample_rate * 0.45
        )

        spectrogram = sound.to_spectrogram(
            window_length=0.005,
            maximum_frequency=max_frequency,
            time_step=0.01,
            frequency_step=25.0
        )

        # 2. 시간, 주파수, 에너지 데이터 추출
        spec_times = spectrogram.xs()
        spec_freqs = spectrogram.ys()
        spec_power = spectrogram.values

        # 3. Power를 dB로 변환
        spec_db = 10 * np.log10(
            np.maximum(spec_power, 1e-12)
        )

        # 화면 표시를 위한 상대 dB
        spec_db = spec_db - np.max(spec_db)
        spec_db = np.maximum(spec_db, -60)

        # 4. Spectrogram 그래프 생성
        fig_select = go.Figure()

        fig_select.add_trace(
            go.Heatmap(
                x=spec_times,
                y=spec_freqs,
                z=spec_db,
                colorscale="Greys",
                reversescale=False,
                zmin=-60,
                zmax=0,
                colorbar=dict(
                    title="Relative dB"
                ),
                name="Spectrogram"
            )
        )

        # 5. 드래그 선택을 지원할 투명 Scatter
        selector_times = spec_times[
            ::max(1, len(spec_times) // 350)
        ]

        selector_freqs = np.linspace(
            0,
            max_frequency,
            30
        )

        selector_x = np.tile(
            selector_times,
            len(selector_freqs)
        )

        selector_y = np.repeat(
            selector_freqs,
            len(selector_times)
        )

        fig_select.add_trace(
            go.Scattergl(
                x=selector_x,
                y=selector_y,
                mode="markers",
                marker=dict(
                    size=8,
                    color="rgba(255,255,255,0.01)"
                ),
                hoverinfo="skip",
                showlegend=False,
                name="Selection Layer"
            )
        )

        # 6. 현재 선택한 시간 범위 표시
        current_start = float(
            st.session_state[start_key]
        )

        current_end = float(
            st.session_state[end_key]
        )

        if current_end > current_start:
            fig_select.add_vrect(
                x0=current_start,
                x1=current_end,
                fillcolor="orange",
                opacity=0.15,
                line_width=0,
                layer="above"
            )

        # 7. 그래프 설정
        fig_select.update_layout(
            title="Select Audio Region - Spectrogram",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            yaxis_range=[0, max_frequency],
            height=450,
            dragmode="select",
            selectdirection="h"
        )

        # 8. 드래그 선택 이벤트
        waveform_event = st.plotly_chart(
            fig_select,
            width="stretch",
            key=chart_key,
            on_select="rerun",
            selection_mode="box"
        )


        # 4. 드래그한 시간 범위 추출
        selected_times = []

        # 박스 선택의 좌우 시간 좌표를 우선 사용
        selected_boxes = waveform_event.selection.box

        if selected_boxes:
            first_box = selected_boxes[-1]

            if "x" in first_box:
                x_range = first_box["x"]

                if len(x_range) == 2:
                    selected_times = [
                        float(x_range[0]),
                        float(x_range[1])
                    ]

        # 박스 좌표를 읽을 수 없을 때 선택 포인트 사용
        if len(selected_times) < 2:
            selected_points = waveform_event.selection.points

            if len(selected_points) >= 2:
                selected_times = [
                    float(point["x"])
                    for point in selected_points
                    if "x" in point
                ]

        # 5. 새 드래그 선택이면 숫자 입력값 갱신
        if len(selected_times) >= 2:
            drag_start = max(
                0.0, min(selected_times)
            )

            drag_end = min(
                float(duration), max(selected_times)
            )

            new_selection = (
                round(drag_start, 6),
                round(drag_end, 6)
            )

            old_selection = st.session_state.get(
                selection_key
            )

            if (
                drag_end > drag_start
                and new_selection != old_selection
            ):
                st.session_state[start_key] = drag_start
                st.session_state[end_key] = drag_end
                st.session_state[selection_key] = new_selection

        # =====================================
        # STEP 17-1. Listen to Selected Region
        # =====================================

        st.subheader("Listen to Selected Region")

        # 현재 선택된 구간의 시작 및 종료 시간
        play_start = float(st.session_state[start_key])
        play_end = float(st.session_state[end_key])

        if 0 <= play_start < play_end <= float(duration):

            # 시간 정보를 실제 오디오 샘플 위치로 변환
            audio_start_idx = max(
                0,
                int(round(play_start * sample_rate))
            )

            audio_end_idx = min(
                sound.n_samples,
                int(round(play_end * sample_rate))
            )

            # 선택 구간의 오디오 추출
            selected_audio = sound.values[
                :, audio_start_idx:audio_end_idx
            ]

            if selected_audio.shape[1] > 0:

                st.audio(
                    selected_audio,
                    sample_rate=int(sample_rate),
                    format="audio/wav"
                )

                st.caption(
                    f"Playback: {play_start:.3f}s "
                    f"~ {play_end:.3f}s"
                )

            else:
                st.warning(
                    "선택된 구간에 재생할 오디오가 없습니다."
                )

        else:
            st.info(
                "먼저 유효한 구간을 선택하세요."
            )

        # 6. 시간 직접 입력
        col_start, col_end = st.columns(2)

        with col_start:
            start_time = st.number_input(
                "Start Time (s)",
                min_value=0.0,
                max_value=float(duration),
                step=0.001,
                format="%.3f",
                key=start_key
            )

        with col_end:
            end_time = st.number_input(
                "End Time (s)",
                min_value=0.0,
                max_value=float(duration),
                step=0.001,
                format="%.3f",
                key=end_key
            )

        # 7. 입력 범위 검증
        if end_time <= start_time:
            st.error(
                "End Time은 Start Time보다 커야 합니다."
            )

        else:


            # 8. 선택 구간 Duration
            region_duration = (
                end_time - start_time
            ) * 1000

            # 9. 선택 구간 F0
            f0_mask = (
                (pitch_times >= start_time)
                & (pitch_times < end_time)
                & np.isfinite(pitch_values)
                & (pitch_values > 0)
            )

            region_f0 = pitch_values[f0_mask]

            region_mean_f0 = (
                float(np.mean(region_f0))
                if len(region_f0) > 0
                else np.nan
            )

            # 10. 선택 구간 Intensity
            intensity_mask = (
                (intensity_times >= start_time)
                & (intensity_times < end_time)
                & np.isfinite(intensity_values)
            )

            region_intensity = intensity_values[
                intensity_mask
            ]

            region_mean_intensity = (
                float(np.mean(region_intensity))
                if len(region_intensity) > 0
                else np.nan
            )


            # 12. 결과표
            region_df = pd.DataFrame([{
                "Start (s)": start_time,
                "End (s)": end_time,
                "Duration (ms)": region_duration,
                "Mean F0 (Hz)": region_mean_f0,
                "Mean Intensity (dB)": region_mean_intensity
            }])

            st.dataframe(
                region_df.round(3),
                hide_index=True,
                width="stretch"
            )

        # =====================================
        # STEP 16. Selected Region Split Analysis
        # =====================================

        st.subheader("Selected Region Spectrogram")

        # 1. 선택된 구간 가져오기
        crop_start = float(st.session_state[start_key])
        crop_end = float(st.session_state[end_key])

        if 0 <= crop_start < crop_end <= float(duration):

            # 2. 선택 구간의 실제 오디오 샘플 추출
            start_idx = max(
                0, int(np.floor(crop_start * sample_rate))
            )

            end_idx = min(
                len(samples),
                int(np.ceil(crop_end * sample_rate))
            )

            cropped_samples = samples[start_idx:end_idx]

            cropped_times = (
                np.arange(start_idx, end_idx)
                / sample_rate
            )

            if len(cropped_samples) > 1:

                st.write(
                    f"Selected: {crop_start:.3f}s "
                    f"~ {crop_end:.3f}s"
                )

                # =====================================
                # Split Position Calculation
                # =====================================

                # 슬라이더의 현재 값을 세션에서 가져오기
                split_fraction = float(
                    st.session_state.get(
                        f"split_fraction_{file_id}",
                        0.50
                    )
                )

                # 그래프 표시 전에 분할 위치 계산
                split_time = (
                    crop_start
                    + (crop_end - crop_start)
                    * split_fraction
                )
                
                # =====================================
                # Full Spectrogram with Split Position
                # =====================================

                fig_split = go.Figure()

                # 전체 오디오 Spectrogram
                fig_split.add_trace(
                    go.Heatmap(
                        x=spec_times,
                        y=spec_freqs,
                        z=spec_db,
                        colorscale="Greys",
                        reversescale=False,
                        zmin=-60,
                        zmax=0,
                        colorbar=dict(
                            title="Relative dB"
                        ),
                        name="Spectrogram"
                    )
                )

                # 선택 구간 시작 경계
                fig_split.add_vline(
                    x=crop_start,
                    line_color="deepskyblue",
                    line_width=2,
                    line_dash="dot"
                )

                # 선택 구간 종료 경계
                fig_split.add_vline(
                    x=crop_end,
                    line_color="deepskyblue",
                    line_width=2,
                    line_dash="dot"
                )

                # Split Position 표시
                fig_split.add_vline(
                    x=split_time,
                    line_color="orange",
                    line_width=2,
                    line_dash="dash"
                )

                # 그래프 설정
                fig_split.update_layout(
                    title="Spectrogram with Split Position",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Frequency (Hz)",
                    xaxis_range=[0, float(duration)],
                    yaxis_range=[0, max_frequency],
                    height=450
                )

                # Spectrogram 출력
                st.plotly_chart(
                    fig_split,
                    width="stretch",
                    key=f"full_split_spectrogram_{file_id}"
                )
                
                # =====================================
                # Split Position Slider
                # =====================================

                split_fraction = st.slider(
                    "Split Position",
                    min_value=0.05,
                    max_value=0.95,
                    value=0.50,
                    step=0.01,
                    key=f"split_fraction_{file_id}"
                )

                st.write(
                    f"Split Time: {split_time:.3f}s"
                )

                # =====================================
                # STEP 17-2. Split Segment Audio Playback
                # =====================================

                st.subheader("Listen to Split Segments")

                # 1. 분리된 두 구간의 시작 및 종료 시간
                audio_regions = [
                    ("Segment 1", crop_start, split_time),
                    ("Segment 2", split_time, crop_end)
                ]

                # 2. 오디오 플레이어를 가로로 배치
                audio_col1, audio_col2 = st.columns(2)

                for i, (name, seg_start, seg_end) in enumerate(
                    audio_regions
                ):

                    # 시간을 실제 오디오 샘플 위치로 변환
                    seg_start_idx = max(
                        0,
                        int(round(seg_start * sample_rate))
                    )

                    seg_end_idx = min(
                        sound.n_samples,
                        int(round(seg_end * sample_rate))
                    )

                    # 원본 WAV에서 구간별 오디오 추출
                    segment_audio = sound.values[
                        :, seg_start_idx:seg_end_idx
                    ]

                    # 왼쪽과 오른쪽 컬럼에 배치
                    column = (
                        audio_col1 if i == 0 else audio_col2
                    )

                    with column:

                        st.write(f"**{name}**")

                        st.caption(
                            f"{seg_start:.3f}s ~ "
                            f"{seg_end:.3f}s"
                        )

                        # 선택한 음절만 재생
                        if segment_audio.shape[1] > 0:

                            st.audio(
                                segment_audio,
                                sample_rate=int(sample_rate),
                                format="audio/wav"
                            )

                        else:
                            st.warning(
                                "재생할 오디오 샘플이 없습니다."
                            )

                # =====================================
                # STEP 19. Manual Segment Selection
                # =====================================

                st.subheader("Selected Region")

                st.write(
                    "음절 개수를 선택한 후, "
                    "스펙트로그램에서 각 음절 구간을 "
                    "순서대로 드래그하여 저장하세요."
                )

                # 1. 선택 구간별로 별도의 상태 관리
                region_id = (
                    f"{file_id}_"
                    f"{int(round(crop_start * 1000))}_"
                    f"{int(round(crop_end * 1000))}"
                )

                # =====================================
                # Number of Syllables
                # =====================================

                segment_count = st.selectbox(
                    "Number of Syllables",
                    options=[2, 3, 4],
                    index=0,
                    key=f"segment_count_{region_id}"
                )

                segments_key = f"manual_segments_{region_id}"
                version_key = f"manual_chart_version_{region_id}"

                if segments_key not in st.session_state:
                    st.session_state[segments_key] = {}

                if version_key not in st.session_state:
                    st.session_state[version_key] = 0

                saved_segments = st.session_state[segments_key]

                # 음절 개수를 줄이면 초과하는 Segment 정리
                for seg_id in list(saved_segments):
                    if seg_id > segment_count:
                        del saved_segments[seg_id]

                # 2. 선택용 Spectrogram
                fig_manual = go.Figure()

                fig_manual.add_trace(
                    go.Heatmap(
                        x=spec_times,
                        y=spec_freqs,
                        z=spec_db,
                        colorscale="Greys",
                        reversescale=False,
                        zmin=-60,
                        zmax=0,
                        colorbar=dict(title="Relative dB")
                    )
                )

                # 전체 시간 범위에서 드래그 선택 지원
                selection_times = spec_times[
                    ::max(1, len(spec_times) // 400)
                ]

                selection_freqs = np.linspace(
                    0, max_frequency, 35
                )

                fig_manual.add_trace(
                    go.Scattergl(
                        x=np.tile(
                            selection_times,
                            len(selection_freqs)
                        ),
                        y=np.repeat(
                            selection_freqs,
                            len(selection_times)
                        ),
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="rgba(255,255,255,0.01)"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

                # 저장된 구간을 그래프에 표시
                segment_colors = {
                    1: "rgba(13,148,136,0.25)",
                    2: "rgba(37,99,235,0.25)",
                    3: "rgba(168,85,247,0.25)",
                    4: "rgba(249,115,22,0.25)"
                }

                for seg_id, seg_range in saved_segments.items():
                    seg_start, seg_end = seg_range

                    fig_manual.add_vrect(
                        x0=seg_start,
                        x1=seg_end,
                        fillcolor=segment_colors[seg_id],
                        line_width=0,
                        layer="above"
                    )

                # =====================================
                # Split Position - Orange Dashed Line
                # =====================================

                fig_manual.add_vline(
                    x=split_time,
                    line_color="orange",
                    line_width=2,
                    line_dash="dash"
                )
                
                # =====================================
                # Selected Audio Region - Boundaries
                # =====================================

                # 선택 구간 시작 위치
                fig_manual.add_vline(
                    x=crop_start,
                    line_color="deepskyblue",
                    line_width=2,
                    line_dash="dot"
                )

                # 선택 구간 종료 위치
                fig_manual.add_vline(
                    x=crop_end,
                    line_color="deepskyblue",
                    line_width=2,
                    line_dash="dot"
                )

                # =====================================
                # X-axis: Fixed 0.05-second Intervals
                # =====================================

                fig_manual.update_layout(
                    title="Select Syllable Regions",

                    xaxis=dict(
                        title="Time (seconds)",
                        range=[0, float(duration)],

                        # 0.05초 간격으로 눈금 고정
                        tickmode="linear",
                        tick0=0,
                        dtick=0.05,
                        tickformat=".2f",

                        # 모든 눈금 라벨 표시
                        tickangle=-45,
                        automargin=True,

                        # 세로 점선 표시
                        showgrid=True,
                        gridcolor="rgba(100,100,100,0.5)",
                        gridwidth=1,
                        griddash="dot",
                        zeroline=False
                    ),

                    yaxis=dict(
                        title="Frequency (Hz)",
                        range=[0, max_frequency]
                    ),

                    height=450,
                    dragmode="select",
                    selectdirection="h"
                )

                # 3. 구간 선택 이벤트
                selection_event = st.plotly_chart(
                    fig_manual,
                    width="stretch",
                    key=(
                        f"manual_selector_{region_id}_"
                        f"{st.session_state[version_key]}"
                    ),
                    on_select="rerun",
                    selection_mode="box"
                )

                # 4. 드래그 범위 읽기
                selected_range = None

                boxes = selection_event.selection.box

                if boxes and "x" in boxes[-1]:
                    x_values = boxes[-1]["x"]

                    if len(x_values) == 2:
                        selected_range = (
                            float(min(x_values)),
                            float(max(x_values))
                        )

                # 박스 정보가 없으면 포인트로 계산
                if selected_range is None:
                    points = selection_event.selection.points

                    point_times = [
                        float(p["x"])
                        for p in points
                        if "x" in p
                    ]

                    if len(point_times) >= 2:
                        selected_range = (
                            min(point_times),
                            max(point_times)
                        )

                # 5. 선택 범위 검증
                valid_selection = False

                if selected_range is not None:
                    selection_start = max(
                        crop_start, selected_range[0]
                    )

                    selection_end = min(
                        crop_end, selected_range[1]
                    )

                    valid_selection = (
                        selection_end > selection_start
                    )

                # 6. 저장할 Segment 결정
                next_segment = None

                for seg_id in range(1, segment_count + 1):
                    if seg_id not in saved_segments:
                        next_segment = seg_id
                        break


                if next_segment is not None:

                    st.write(
                        f"**Selecting Segment {next_segment}**"
                    )

                    if valid_selection:

                        st.write(
                            f"Selected: "
                            f"{selection_start:.3f}s ~ "
                            f"{selection_end:.3f}s"
                        )

                        # Segment 2는 Segment 1 이후에 위치
                        if next_segment > 1:
                            previous_end = saved_segments[
                                next_segment - 1
                            ][1]

                            if selection_start < previous_end:
                                valid_selection = False
                                st.warning(
                                    f"Segment {next_segment}는 "
                                    f"Segment {next_segment - 1}이 "
                                    "끝난 뒤에 선택하세요."
                                )


                    # =====================================
                    # Listen to Preview Before Saving
                    # =====================================

                    if valid_selection:

                        st.write("**Listen to Preview**")

                        # 선택 시간을 오디오 샘플 위치로 변환
                        preview_start_sample = max(
                            0,
                            int(round(selection_start * sample_rate))
                        )

                        preview_end_sample = min(
                            sound.n_samples,
                            int(round(selection_end * sample_rate))
                        )

                        # 원본 오디오에서 선택 구간 추출
                        preview_audio = sound.values[
                            :,
                            preview_start_sample:preview_end_sample
                        ]

                        # 선택한 구간의 길이 표시
                        preview_duration = (
                            preview_audio.shape[1] / sample_rate
                        )

                        st.caption(
                            f"Preview Duration: "
                            f"{preview_duration:.3f} s"
                        )

                        # 저장 전 오디오 재생
                        if preview_audio.shape[1] > 0:

                            st.audio(
                                preview_audio,
                                sample_rate=int(sample_rate),
                                format="audio/wav"
                            )

                        else:
                            st.warning(
                                "선택한 구간에 재생할 "
                                "오디오 샘플이 없습니다."
                            )

                    if st.button(
                        f"Save as Segment {next_segment}",
                        disabled=not valid_selection,
                        key=f"save_manual_{region_id}"
                    ):

                        saved_segments[next_segment] = (
                            selection_start,
                            selection_end
                        )

                        st.session_state[segments_key] = (
                            saved_segments
                        )

                        # 다음 선택을 위해 그래프 상태 초기화
                        st.session_state[version_key] += 1
                        st.rerun()

                else:
                    st.success(
                        f"Segment 1~{segment_count}이 모두 저장되었습니다."                        
                    )

                # 7. 선택 초기화
                if st.button(
                    "Reset Selected Segments",
                    key=f"reset_manual_{region_id}"
                ):
                    st.session_state[segments_key] = {}
                    st.session_state[version_key] += 1
                    st.rerun()

                # 8. 저장된 Segment별 오디오와 음향 분석
                manual_results = []

                if saved_segments:
                    st.subheader("Listen to Selected Segments")

                    audio_columns = st.columns(2)

                    for seg_id in sorted(saved_segments):
                        seg_start, seg_end = saved_segments[seg_id]

                        start_sample = max(
                            0, int(round(seg_start * sample_rate))
                        )

                        end_sample = min(
                            sound.n_samples,
                            int(round(seg_end * sample_rate))
                        )

                        seg_audio = sound.values[
                            :, start_sample:end_sample
                        ]

                        with audio_columns[(seg_id - 1) % 2]:
                            st.write(f"**Segment {seg_id}**")

                            st.caption(
                                f"{seg_start:.3f}s ~ "
                                f"{seg_end:.3f}s"
                            )

                            if seg_audio.shape[1] > 0:
                                st.audio(
                                    seg_audio,
                                    sample_rate=int(sample_rate),
                                    format="audio/wav"
                                )


                        # F0
                        f0_mask = (
                            (pitch_times >= seg_start)
                            & (pitch_times < seg_end)
                            & np.isfinite(pitch_values)
                            & (pitch_values > 0)
                        )

                        seg_f0 = pitch_values[f0_mask]

                        mean_f0 = (
                            float(np.mean(seg_f0))
                            if len(seg_f0) > 0
                            else np.nan
                        )

                        # Intensity
                        db_mask = (
                            (intensity_times >= seg_start)
                            & (intensity_times < seg_end)
                            & np.isfinite(intensity_values)
                        )

                        seg_db = intensity_values[db_mask]

                        mean_db = (
                            float(np.mean(seg_db))
                            if len(seg_db) > 0
                            else np.nan
                        )

                        # 결과 저장
                        manual_results.append({
                            "Segment": f"Segment {seg_id}",
                            "Start (s)": seg_start,
                            "End (s)": seg_end,
                            "Duration (ms)":
                                (seg_end - seg_start) * 1000,
                            "Mean F0 (Hz)": mean_f0,
                            "Mean Intensity (dB)": mean_db
                        })

                # 9. 분석 결과 및 CSV 다운로드
                if manual_results:

                    manual_df = pd.DataFrame(
                        manual_results
                    )

                    st.subheader("Split Segment Results")

                    st.dataframe(
                        manual_df.round(3),
                        hide_index=True,
                        width="stretch"
                    )

                    if len(manual_df) == segment_count:

                        manual_csv = manual_df.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            label="Download Split Results",
                            data=manual_csv,
                            file_name="Manual_Split_Results.csv",
                            mime="text/csv",
                            key=f"download_manual_{region_id}"
                        )
                    else:
                        st.info(
                            f"Segment {segment_count}까지 선택하면 "
                            "CSV를 다운로드할 수 있습니다."                            
                        )

            else:
                st.warning(
                    "선택된 구간에 오디오 샘플이 부족합니다."
                )
        else:
            st.info(
                "먼저 유효한 오디오 구간을 선택하세요."
            )

    except Exception as error:
        st.error("음성 파일을 분석하지 못했습니다.")
        st.exception(error)

    finally:
        # 11. 임시 파일 삭제
        if temp_path is not None and os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("Please upload a WAV audio file.")

# ==========================================
# STEP 13. Multiple Audio File Analysis
# ==========================================

st.divider()
st.header("Batch Audio Analysis")

st.write(
    "Upload multiple WAV files to compare "
    "F0, Intensity, and Duration."
)

# 1. 여러 WAV 파일 업로드
batch_files = st.file_uploader(
    "Upload multiple WAV files",
    type=["wav"],
    accept_multiple_files=True,
    key="batch_audio_uploader"
)

if batch_files:

    st.write(f"Uploaded files: {len(batch_files)}")

    batch_results = []

    # 2. 파일별 분석
    for file_number, audio_file in enumerate(
        batch_files, start=1
    ):

        batch_temp_path = None

        try:
            # 임시 WAV 파일 저장
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:

                temp_file.write(audio_file.getvalue())
                batch_temp_path = temp_file.name

            # 3. Parselmouth로 음성 파일 읽기
            batch_sound = parselmouth.Sound(
                batch_temp_path
            )

            # 전체 녹음 길이
            file_duration = batch_sound.duration

            # 4. F0 분석
            batch_pitch = batch_sound.to_pitch(
                time_step=0.01,
                pitch_floor=75,
                pitch_ceiling=600
            )

            f0_values = batch_pitch.selected_array[
                "frequency"
            ]

            valid_f0 = f0_values[
                np.isfinite(f0_values)
                & (f0_values > 0)
            ]

            if len(valid_f0) > 0:
                file_mean_f0 = float(
                    np.mean(valid_f0)
                )
            else:
                file_mean_f0 = np.nan

            # 5. Intensity 분석
            batch_intensity = batch_sound.to_intensity(
                minimum_pitch=100,
                time_step=0.01
            )

            db_values = batch_intensity.values[0]

            valid_db = db_values[
                np.isfinite(db_values)
            ]

            if len(valid_db) > 0:
                file_mean_intensity = float(
                    np.mean(valid_db)
                )
            else:
                file_mean_intensity = np.nan

            # 6. 파일별 결과 저장
            batch_results.append({
                "File No.": file_number,
                "File Name": audio_file.name,
                "Duration (s)": file_duration,
                "Mean F0 (Hz)": file_mean_f0,
                "Mean Intensity (dB)": file_mean_intensity
            })

        except Exception as error:
            st.error(
                f"{audio_file.name} 분석 실패: {error}"
            )

        finally:
            # 임시 파일 삭제
            if (
                batch_temp_path is not None
                and os.path.exists(batch_temp_path)
            ):
                os.remove(batch_temp_path)

    # 7. 분석 결과 출력
    if batch_results:

        batch_df = pd.DataFrame(
            batch_results
        )

        st.subheader("Individual File Results")

        st.dataframe(
            batch_df.round(2),
            hide_index=True,
            width="stretch"
        )

        # 8. 전체 평균 계산
        overall_mean_f0 = batch_df[
            "Mean F0 (Hz)"
        ].mean()

        overall_mean_intensity = batch_df[
            "Mean Intensity (dB)"
        ].mean()

        overall_mean_duration = batch_df[
            "Duration (s)"
        ].mean()

        st.subheader("Overall Average")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Overall Mean F0",
            (
                f"{overall_mean_f0:.2f} Hz"
                if pd.notna(overall_mean_f0)
                else "N/A"
            )
        )

        col2.metric(
            "Overall Mean Intensity",
            (
                f"{overall_mean_intensity:.2f} dB"
                if pd.notna(overall_mean_intensity)
                else "N/A"
            )
        )

        col3.metric(
            "Average Duration",
            f"{overall_mean_duration:.2f} s"
        )

        # 9. 전체 평균 결과표
        overall_df = pd.DataFrame([{
            "Analyzed Files": len(batch_df),
            "Overall Mean F0 (Hz)": overall_mean_f0,
            "Overall Mean Intensity (dB)":
                overall_mean_intensity,
            "Average Duration (s)":
                overall_mean_duration
        }])

        # 10. 파일별 F0 비교 그래프
        st.subheader("Mean F0 by File")

        fig_batch_f0 = go.Figure()

        fig_batch_f0.add_trace(
            go.Bar(
                x=[
                    f"{i}: {name}"
                    for i, name in zip(
                        batch_df["File No."],
                        batch_df["File Name"]
                    )
                ],
                y=batch_df["Mean F0 (Hz)"],
                name="Mean F0"
            )
        )

        fig_batch_f0.update_layout(
            xaxis_title="Audio File",
            yaxis_title="Mean F0 (Hz)",
            height=400
        )

        st.plotly_chart(
            fig_batch_f0,
            width="stretch"
        )

        # 11. CSV 다운로드
        st.subheader("Download Batch Results")

        file_csv = batch_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        overall_csv = overall_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download Individual Results",
                data=file_csv,
                file_name="Batch_Individual_Results.csv",
                mime="text/csv",
                key="batch_individual_download"
            )

        with col2:
            st.download_button(
                label="Download Overall Average",
                data=overall_csv,
                file_name="Batch_Overall_Average.csv",
                mime="text/csv",
                key="batch_average_download"
            )

    else:
        st.warning(
            "분석 가능한 WAV 파일이 없습니다."
        )
