# -*- coding: utf-8 -*-

from os import pathconf_names
from random import randint
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import seaborn as sns
import plotly.express as px

import copy

import cv2
from PIL import Image, ImageDraw

from streamlit_image_coordinates import streamlit_image_coordinates

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
###################################

if "points" not in st.session_state:
    st.session_state["points"] = []

if 'clear_falg' not in st.session_state:
    st.session_state["clear_falg"] = 0

def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )

## マスク画像を統合（複数のマスクを１つに統合）
def merge_masks(mask_images):
    
    if len(mask_images) == 1:
        return mask_images[0]
    else:
        merged = np.zeros(mask_images[0].shape, dtype='uint8')

        for m in mask_images:
            merged = cv2.bitwise_or(merged, m)
            #merged = np.bitwise_or(merged, m)

        return merged


def main():
    # ヘッダ。　ロゴとシステム名
    col_logo, col_pagetitle = st.columns([3,13])

    with col_logo:
        st.write("  ")
        st.write("  ")
        img_logo = Image.open("dronepilot_logo.png")
        st.image(img_logo)

    with col_pagetitle:
        st.title('DPA Report')

    #### 処理本体

    #filename = "sabi_001.png"
    #filename = "s_target.jpg"

    ## 画像ファイルの読み込み
    #img = cv2.imread(filename)
    #pil_img = cv2.imread(filename)
    #mask_img = np.zeros(img[:,:,0].shape) ## マスク画像（1ch）を準備しておく

    ### 外部ファイルを選択する場合
    if st.checkbox("分析するファイルを選択"):
        uploaded = st.file_uploader("解析対象の画像をアップロード", type=['png', 'jpg'])
        if uploaded:
            pil_img = Image.open(uploaded)    # 基本的にはPILのImageでしか読めない
            img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    ##### カラム 設定
    col_setting, col_img_src = st.columns([2, 15])
    col_noSetting, col_button = st.columns([2, 15])

    with col_setting:
        '''
        #### 設定
        '''
        #num_class = st.slider("色の数", min_value=2, max_value=50, step=1, value=12)
        # num_k_iteration = st.slider("精度(分析回数)", min_value=3, max_value=30, step=1, value=6)
        num_k_iteration = 4 # ひとまず決め打ち
        alpha_percent = st.slider("強調度", min_value=0, max_value=100, step=1, value=65)

    ### クラスタリングの実行

    # 画像で使用されている色一覧。(W * H, 3) の numpy 配列。
    colors = img.reshape(-1, 3).astype(np.float32)

    ## クラスタリング関係
    # 最大反復回数: 10、移動量の閾値: 1.0
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, num_k_iteration, 1.0

    # num_class (max 50 で設定)
    ret, labels, centers = cv2.kmeans(
        colors, 50, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.squeeze(axis=1)  # (N, 1) -> (N,)
    centers = centers.astype(np.uint8)  # float32 -> uint8
    
    # 減色後の画像
    dst = centers[labels].reshape(img.shape)
    
    with col_img_src:

        st.subheader('解析対象画像')

        #画像のサイズ変更
        height = img.shape[0]
        width = img.shape[1]
        width = round(width * 400 / height)
        resized_img = cv2.resize(img,(width, 400),interpolation = cv2.INTER_CUBIC)
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        new_img = Image.fromarray(resized_img)

        draw = ImageDraw.Draw(new_img)

        list_RGB = [] # 選択された色一覧を格納

        # Draw an ellipse at each coordinate in points
        for point in st.session_state["points"]:

            c_R,c_G,c_B = new_img.getpixel(point)
            #st.text(c_R + c_G + c_B)
            # RGB の保存 
            list_RGB.append([c_R, c_G, c_B])
            coords = get_ellipse_coords(point)
            draw.ellipse(coords, fill="red")

        if st.session_state["clear_falg"] == 1:
            value = None
            st.session_state["clear_falg"] = 0
            st.experimental_rerun()
        else:
            value = streamlit_image_coordinates(new_img, key="pil")

        if value is not None:
            point = value["x"], value["y"]
                        
            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                st.experimental_rerun()

    with col_button:
        #col = st.columns(1)
        col1, col2, col3, col4 = st.columns(4)
        
        show_button = col1.button('サビを検出')
        clear_button = col2.button('クリア')

        if clear_button:
            #del st.session_state["points"]
            if len(st.session_state["points"]) > 0:
                st.session_state["clear_falg"] = 1
                del st.session_state["points"]
                st.experimental_rerun()

        if show_button:

            st.subheader('解析結果の表示')

            mask_imgs = []
            # RGB からマスク画像を作成
            for rgb in list_RGB:

                c_R,c_G,c_B = rgb
                
                # 画素値の上限・下限を決める
                p_delta = 24        # 16と32の間
                lower = (max(c_B - p_delta, 0), max(c_G - p_delta, 0), max(c_R - p_delta, 0))
                upper = (min(c_B + p_delta, 255), min(c_G + p_delta, 255), min(c_R + p_delta, 255))

                bin_img = cv2.inRange(img, lower, upper)
                mask_imgs.append(bin_img)
                mask_img = merge_masks(mask_imgs)

            mask_5ch_img = copy.deepcopy(img)
            mask_5ch_img[:,:,(1,2)] = 0  
            mask_5ch_img[:,:,0] = mask_img

            # 透過率に基づいてブレンディング
            alpha = alpha_percent / 100.0
            marked_img = cv2.addWeighted(dst, (1-alpha), mask_5ch_img, alpha, 2.2)

            st.image(marked_img, "強調画像", use_column_width=True, channels="BGR")

    # フッタの処理
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    main()

### cf.
# [1]: https://pystyle.info/opencv-kmeans/
###

####
# End of This File 
####
