import numpy as np
import cv2
import matplotlib.pyplot as plt

def RegionSegWithSLIC(img, n_segments=50, resolution=0.050000, origin=[-22.800000, -50.000000, 0.000000], min_area=500):
    height, width = img.shape[:2]
    region_size = int(np.sqrt((width * height) / n_segments))
    img[img >= 250] = 255

    # SLIC 생성 및 Superpixel 계산
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=region_size, ruler=20)
    slic.iterate(10)

    # Superpixel 레이블과 경계선 마스크 가져오기
    labels = slic.getLabels()
    mask = slic.getLabelContourMask()

    img_with_contours = img.copy()
    img_with_contours[mask == 255] = [0, 255, 0]
    
    # 결과 시각화
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('SLIC Superpixel Segmentation')
    plt.show()
    
    img_color = img.copy()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_mask = gray_img == 255

    unique_segments = np.unique(labels)

    # 각 Superpixel에 대해 흰색 영역을 가진 부분만 컬러링
    for seg_id in unique_segments:
        mask_seg = (labels == seg_id) & white_mask
        if np.any(mask_seg):
            color = np.random.randint(0, 255, size=3)
            img_color[mask_seg] = color

    img_color[mask == 255] = [0, 0, 255]

    # 결과 시각화
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title('SLIC Superpixel Segmentation with White Regions Colored (OpenCV)')
    plt.show()

    unique_segments = np.unique(labels)
    img_color = img.copy()
    polygon_list_robot = []
    polygon_list_map = []
    centroid_list = []  # 중심 좌표 리스트 추가
    
    shape_list = []

    for idx, seg_id in enumerate(unique_segments):
        mask_seg = (labels == seg_id) & white_mask  
        if np.any(mask_seg):
            mask_seg_uint8 = mask_seg.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue  # 면적이 작은 세그먼트는 무시하고 제거
                
                contour_coords = []
                contour_map_coords = []
                for point in contour:
                    map_x, map_y = point[0][0], point[0][1]
                    world_x = origin[0] + map_x * resolution
                    world_y = origin[1] + map_y * resolution
                    contour_coords.append({"x": str(world_x), "y": str(world_y)})
                    contour_map_coords.append([map_x, map_y])

                polygon_list_map.append(contour_map_coords)

                # 중심 좌표 계산 (centroid)
                M = cv2.moments(contour)
                if M["m00"] != 0:  # 0으로 나누기 방지
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    centroid_list.append([centroid_x, centroid_y])

    # Segmentation 결과 시각화 (폴리곤과 중심점 함께 표시)
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    for polygon in polygon_list_map:
        polygon = np.array(polygon)
        plt.plot(polygon[:, 0], polygon[:, 1], linewidth=2)
    
    # 중심 좌표 시각화
    for centroid in centroid_list:
        plt.plot(centroid[0], centroid[1], 'ro', markersize=5)  # 빨간 점으로 표시
    plt.title('Segmented Map with Filtered Small Regions and Centroids')
    plt.show()
    
    img_with_polygons = img.copy()
    for polygon in polygon_list_map:
        polygon = np.array(polygon, dtype=np.int32)
        color = np.random.randint(0, 255, size=(3,)).tolist()  # 임의의 색상 선택
        cv2.fillPoly(img_with_polygons, [polygon], color=color)  # 폴리곤 내부를 색칠함

    # 중심점을 이미지에 추가
    for centroid in centroid_list:
        cv2.circle(img_with_polygons, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)  # 빨간 원으로 표시

    # 이미지 저장
    cv2.imwrite('segmented_map_skku_with_centroids.png', img_with_polygons)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_with_polygons, cv2.COLOR_BGR2RGB))
    plt.title('Saved Image with Centroids')
    plt.show()

# 실행
img = cv2.imread("./skku7th.pgm")    
RegionSegWithSLIC(img, n_segments=80)