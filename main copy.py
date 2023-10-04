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


## RGBの値から輝度値Yを計算
def calc_Y(val_RGB):
    return int(round(0.299 * val_RGB[0] + 0.587 * val_RGB[1] + 0.114 * val_RGB[2],0))

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

if "points" not in st.session_state:
    st.session_state["points"] = []

def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )

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
#    filename = "sabi_001.png"
    filename = "s_target.jpg"

    ## 画像ファイルの読み込み
    img = cv2.imread(filename)
    img2 = cv2.imread(filename)
    mask_img = np.zeros(img[:,:,0].shape) ## マスク画像（1ch）を準備しておく

    ### 外部ファイルを選択する場合
    # 【TBD】ファイルが指定されたかどうかの管理
    
    if 'count' not in st.session_state:
      st.session_state["count"] = 0
    
    if st.checkbox("分析するファイルを選択"):
        uploaded = st.file_uploader("ファイルアップロード", type=['png', 'jpg'])
        if uploaded:
            img2 = Image.open(uploaded)    # 基本的にはPILのImageでしか読めない
            pil_img = Image.open(uploaded)    # 基本的にはPILのImageでしか読めない
            img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)       

    list_RGB = [] # 選択された色一覧を格納
    
    ##### カラム 設定
    col_setting, col_img_src, col_table = st.columns([2, 7, 4])

    with col_setting:
        '''
        #### 設定
        '''
        num_class = st.slider("色の数", min_value=2, max_value=50, step=1, value=12)
        # num_k_iteration = st.slider("精度(分析回数)", min_value=3, max_value=30, step=1, value=6)
        num_k_iteration = 4 # ひとまず決め打ち
        alpha_percent = st.slider("強調度", min_value=0, max_value=100, step=1, value=65)

    ### クラスタリングの実行

    # 画像で使用されている色一覧。(W * H, 3) の numpy 配列。
    colors = img.reshape(-1, 3).astype(np.float32)

    ## クラスタリング関係
    # 最大反復回数: 10、移動量の閾値: 1.0
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, num_k_iteration, 1.0

    # 
    ret, labels, centers = cv2.kmeans(
        colors, num_class, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.squeeze(axis=1)  # (N, 1) -> (N,)
    centers = centers.astype(np.uint8)  # float32 -> uint8

    color_codes = []
    for c in centers:
        color_code = '#{:02x}{:02x}{:02x}'.format(c[2], c[1], c[0])
        color_code = color_code.replace('0x', '')
        color_codes.append(color_code)

    # 輝度値【参考】を計算    
    Y = []
    for c in centers:
        Y.append(calc_Y(c))

    ## 描画部分
    # 各クラスタに属するサンプル数を計算する。
    _, counts = np.unique(labels, axis=0, return_counts=True)
    
    # 減色後の画像
    dst = centers[labels].reshape(img.shape)

    with col_img_src:
        st.image(img, "解析対象画像", use_column_width=True, channels="BGR")

        width = round(img2.width * 400 / img2.height)
        new_img = img2.resize((width, 400))

        draw = ImageDraw.Draw(new_img)
        
        mask_imgs = []
        # Draw an ellipse at each coordinate in points
        for point in st.session_state["points"]:

            #st.write(point)
            c_R,c_G,c_B = new_img.getpixel(point)
            st.write(c_R, c_G, c_B)

            # 画素値の上限・下限を決める
            p_delta = 24        # 16と32の間
            lower = (max(c_B - p_delta, 0), max(c_G - p_delta, 0), max(c_R - p_delta, 0))
            upper = (min(c_B + p_delta, 255), min(c_G + p_delta, 255), min(c_R + p_delta, 255))

            bin_img = cv2.inRange(img, lower, upper)

            mask_imgs.append(bin_img)

            mask_img = merge_masks(mask_imgs)

            # mask_4ch_img = copy.deepcopy(img)

            # mask_4ch_img[:,:,(1,2)] = 0          # BGRの「GR」に、0を代入
            # mask_4ch_img[:,:,0] = mask_imgs[0]   # BGRの「B」に、mask_imgの値を代入

            mask_5ch_img = copy.deepcopy(img)

            mask_5ch_img[:,:,(1,2)] = 0  
            mask_5ch_img[:,:,0] = mask_img

            #st.image(mask_img)
            #st.image(mask_5ch_img)

            # 透過率に基づいてブレンディング
            alpha = alpha_percent / 100.0
            marked_img = cv2.addWeighted(dst, (1-alpha), mask_5ch_img, alpha, 2.2)
            # marked_img = cv2.addWeighted(dst, (1-alpha), mask_4ch_img, alpha, 2.2)

            st.image(marked_img, "強調画像", use_column_width=True, channels="BGR")


            coords = get_ellipse_coords(point)
            draw.ellipse(coords, fill="red")


        value = streamlit_image_coordinates(new_img, key="pil")

        if value is not None:
            point = value["x"], value["y"]
            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                st.experimental_rerun()

    with col_table:
        # 表の準備
        df = pd.DataFrame()

        df['RGB'] = color_codes
        df['hist'] = counts
        df['明るさ【参考】'] = Y

        ## チェックボックス付きの表
        from st_aggrid import GridUpdateMode, DataReturnMode

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)

        cellstyle_jscode = JsCode("""
            function(params){
                if (params.value.match(/^#(?:[0-9a-fA-F]{3}){1,2}$/g)) {
                    var color = params.value.substring(1, 7);
                    var r = parseInt(color.substring(0, 2), 16);
                    var g = parseInt(color.substring(2, 4), 16);
                    var b = parseInt(color.substring(4, 6), 16);
                    var textColor = (((r * 0.299) + (g * 0.587) + (b * 0.114)) < 156) ? "#FFFFFF" : "#000000";
                        return {
                        'backgroundColor': params.value,
                        'color': textColor,
                    }
                }
            }
            """)
        gb.configure_columns('RGB', cellStyle= cellstyle_jscode)  ###

        gridOptions = gb.build()

        response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True
        )

        # 選択された行のみ取り出し
        selected_rows = response["selected_rows"]
        df_sel = pd.DataFrame(selected_rows)
        # st.dataframe(df_sel)

        # 対象のピクセル数をカウント
        px_num_target = 0
        if "hist" in df_sel:
            list_target = df_sel["hist"]
            for n in list_target:
                px_num_target += n

            # 指定したRGB値の色を変更  # TBD
            list_RGB = df_sel['RGB']


    ## 損傷率（%）の表示
    i_w, i_h, _ = img.shape
    total_px = i_w * i_h
    t_per = round(100 * px_num_target / total_px, 1)
    st.write("サビ: " + str(px_num_target) + "ピクセル" + " (" + str(t_per) + "%)")

    mask_imgs = []
    # リストで選択した色を、もとの画像で強調して表示
    for c in list_RGB:
        c_R = int("0x" + c[1:3], 16)
        c_G = int("0x" + c[3:5], 16)
        c_B = int("0x" + c[5:7], 16)
        
        st.write(c)
        st.write(c_R, c_G, c_B)

        # 画素値の上限・下限を決める
        p_delta = 24        # 16と32の間
        lower = (max(c_B - p_delta, 0), max(c_G - p_delta, 0), max(c_R - p_delta, 0))
        upper = (min(c_B + p_delta, 255), min(c_G + p_delta, 255), min(c_R + p_delta, 255))

        bin_img = cv2.inRange(img, lower, upper)

        mask_imgs.append(bin_img)
        # mask_img = cv2.bitwise_or(mask_img, bin_img) #src1, src2

    ## チェックボックスがONのときだけ 動く
    if len(selected_rows) > 0:
        mask_img = merge_masks(mask_imgs)

        # mask_4ch_img = copy.deepcopy(img)

        # mask_4ch_img[:,:,(1,2)] = 0          # BGRの「GR」に、0を代入
        # mask_4ch_img[:,:,0] = mask_imgs[0]   # BGRの「B」に、mask_imgの値を代入

        mask_5ch_img = copy.deepcopy(img)

        mask_5ch_img[:,:,(1,2)] = 0  
        mask_5ch_img[:,:,0] = mask_img

        #st.image(mask_img)
        #st.image(mask_5ch_img)

        # 透過率に基づいてブレンディング
        alpha = alpha_percent / 100.0
        marked_img = cv2.addWeighted(dst, (1-alpha), mask_5ch_img, alpha, 2.2)
        # marked_img = cv2.addWeighted(dst, (1-alpha), mask_4ch_img, alpha, 2.2)

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
