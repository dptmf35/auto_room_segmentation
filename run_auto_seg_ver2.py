import numpy as np
import cv2
import argparse
import json
import matplotlib.pyplot as plt

class RoomSegmentation:
    def __init__(self, img_path, n_segments=35, min_area=500, output_file="output_map.png", 
                 json_output="centroids.json", slic_iter=100, ruler=100):
        self.img_path = img_path
        self.n_segments = n_segments
        self.min_area = min_area
        self.output_file = output_file
        self.json_output = json_output
        self.slic_iter = slic_iter
        self.ruler = ruler
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"cannot find image file: {img_path}")
        
        # variables for saving results
        self.labels = None
        self.contour_mask = None
        self.polygon_list_map = []
        self.centroid_list = []
        self.white_mask = None
    
    def preprocess_image(self):
        """전처리: 흰색 영역(맵 내부) 추출"""
        # 이미지에서 흰색 영역(맵 내부) 추출
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) == 3 else self.img
        self.white_mask = gray_img >= 250  # 흰색 영역 마스크
        white_mask_uint8 = self.white_mask.astype(np.uint8) * 255

        # 마스크 확장 (선택적) - 경계 부분을 포함하기 위해
        kernel = np.ones((3, 3), np.uint8)
        white_mask_dilated = cv2.dilate(white_mask_uint8, kernel, iterations=1)
        self.white_mask = white_mask_dilated > 0
        
        return self.white_mask
    
    def create_masked_image(self):
        """흰색 영역만 포함하는 마스킹된 이미지 생성"""
        # 원본 이미지 복사
        img_masked = self.img.copy()
        
        # 맵 외부(흰색이 아닌 부분)를 검은색으로 설정
        if len(self.img.shape) == 3:
            for i in range(3):
                img_masked[:, :, i] = np.where(self.white_mask, self.img[:, :, i], 0)
        else:
            img_masked = np.where(self.white_mask, self.img, 0)
            
        return img_masked
    
    def apply_slic(self):
        """SLIC 알고리즘 적용"""
        # 마스킹된 이미지 생성
        img_masked = self.create_masked_image()
        
        # SLIC 파라미터 계산
        height, width = self.img.shape[:2]
        region_size = int(np.sqrt((width * height) / self.n_segments))
        
        # SLIC 알고리즘 적용 (마스킹된 이미지에만)
        slic = cv2.ximgproc.createSuperpixelSLIC(img_masked, region_size=region_size, ruler=self.ruler)
        slic.iterate(self.slic_iter)
        
        # Superpixel 레이블과 경계선 마스크 가져오기
        self.labels = slic.getLabels()
        self.contour_mask = slic.getLabelContourMask()
        
        # 맵 외부 영역의 레이블을 -1로 설정 (무시하기 위해)
        self.labels = np.where(self.white_mask, self.labels, -1)
        
        return self.labels, self.contour_mask
    
    def create_contour_image(self):
        """경계선이 표시된 이미지 생성"""
        img_with_contours = self.img.copy()
        img_with_contours[self.contour_mask == 255] = [0, 255, 0]  # 경계선을 녹색으로 표시
        
        # 맵 외부 영역은 표시하지 않음
        if len(self.img.shape) == 3:
            for i in range(3):
                img_with_contours[:, :, i] = np.where(self.white_mask, img_with_contours[:, :, i], self.img[:, :, i])
        else:
            # 그레이스케일 이미지인 경우
            img_with_contours = np.where(self.white_mask, img_with_contours, self.img)
            
        return img_with_contours
    
    def create_colored_segments(self):
        """색상이 입혀진 세그먼트 이미지 생성"""
        img_color = self.img.copy()
        unique_segments = np.unique(self.labels)
        # -1 제외 (맵 외부)
        unique_segments = unique_segments[unique_segments >= 0]
        
        for seg_id in unique_segments:
            mask_seg = (self.labels == seg_id)
            if np.any(mask_seg):
                color = np.random.randint(0, 255, size=3)
                if len(img_color.shape) == 3:
                    img_color[mask_seg] = color
                else:
                    # 그레이스케일 이미지인 경우 컬러로 변환 필요
                    img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
                    img_color[mask_seg] = color
        
        # 경계선 강조
        img_color[self.contour_mask == 255] = [0, 0, 255]  # 경계선을 빨간색으로 표시
        
        # 맵 외부 영역을 원본 이미지로 유지
        if len(self.img.shape) == 3:
            for i in range(3):
                img_color[:, :, i] = np.where(self.white_mask, img_color[:, :, i], self.img[:, :, i])
        else:
            img_color = np.where(self.white_mask, img_color, self.img)
            
        return img_color
    
    def extract_polygons(self):
        """세그먼트에서 폴리곤 추출"""
        unique_segments = np.unique(self.labels)
        # -1 제외 (맵 외부)
        unique_segments = unique_segments[unique_segments >= 0]
        
        self.polygon_list_map = []
        self.centroid_list = []
        
        for seg_id in unique_segments:
            mask_seg = (self.labels == seg_id) & self.white_mask
            if np.any(mask_seg):
                # 마스크를 uint8로 변환
                mask_seg_uint8 = mask_seg.astype(np.uint8) * 255
                
                # 외곽선 찾기
                contours, _ = cv2.findContours(mask_seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # 면적 계산
                    area = cv2.contourArea(contour)
                    
                    # 최소 영역 크기보다 작은 세그먼트는 무시
                    if area < self.min_area:
                        continue
                    
                    # 폴리곤 좌표 추출
                    contour_map_coords = []
                    for point in contour:
                        map_x, map_y = point[0][0], point[0][1]
                        contour_map_coords.append([map_x, map_y])
                    
                    self.polygon_list_map.append(contour_map_coords)
                    
                    # 센트로이드(중심점) 계산 - 모멘트 방법 사용
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])
                        
                        # 센트로이드가 폴리곤 외부에 있는 경우 거리 변환 방법 사용
                        if cv2.pointPolygonTest(contour, (centroid_x, centroid_y), False) < 0:
                            # 거리 변환 방법 사용
                            mask_contour = np.zeros_like(mask_seg_uint8)
                            cv2.drawContours(mask_contour, [contour], 0, 255, -1)
                            
                            # 거리 변환 적용
                            dist_transform = cv2.distanceTransform(mask_contour, cv2.DIST_L2, 5)
                            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                            
                            # 최대 거리 지점을 센트로이드로 사용
                            centroid_x, centroid_y = max_loc
                            
                            # 여전히 외부에 있는 경우 폴리곤 내부의 임의의 점 선택
                            if cv2.pointPolygonTest(contour, (centroid_x, centroid_y), False) < 0:
                                # 폴리곤 내부의 점 샘플링
                                x, y, w, h = cv2.boundingRect(contour)
                                found = False
                                for _ in range(100):  # 최대 100번 시도
                                    sample_x = np.random.randint(x, x + w)
                                    sample_y = np.random.randint(y, y + h)
                                    if cv2.pointPolygonTest(contour, (sample_x, sample_y), False) >= 0:
                                        centroid_x, centroid_y = sample_x, sample_y
                                        found = True
                                        break
                                
                                # 찾지 못한 경우 폴리곤의 첫 번째 점 사용
                                if not found:
                                    centroid_x, centroid_y = contour[0][0]
                    else:
                        # 모멘트가 0인 경우(드문 경우) 폴리곤의 첫 번째 점 사용
                        centroid_x, centroid_y = contour[0][0]
                    
                    self.centroid_list.append([centroid_x, centroid_y])
        
        # 폴리곤 및 센트로이드 개수 출력
        num_segments = len(self.polygon_list_map)
        print(f"생성된 폴리곤(세그먼트) 개수: {num_segments}")
        
        num_centroids = len(self.centroid_list)
        print(f"계산된 센트로이드 개수: {num_centroids}")
        
        return self.polygon_list_map, self.centroid_list
    
    def create_final_image(self):
        """최종 이미지 생성 (폴리곤과 센트로이드 포함)"""
        img_with_polygons = self.img.copy()
        if len(img_with_polygons.shape) == 2:
            img_with_polygons = cv2.cvtColor(img_with_polygons, cv2.COLOR_GRAY2BGR)
        
        # 폴리곤 채우기
        for polygon in self.polygon_list_map:
            polygon = np.array(polygon, dtype=np.int32)
            color = np.random.randint(0, 255, size=(3,)).tolist()
            cv2.fillPoly(img_with_polygons, [polygon], color=color)
        
        # 센트로이드 추가
        for centroid in self.centroid_list:
            cv2.circle(img_with_polygons, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)
            
        return img_with_polygons
    
    def save_result(self, img):
        """결과 이미지 저장"""
        cv2.imwrite(self.output_file, img)
        print(f"결과 이미지 저장: {self.output_file}")
    
    def save_centroids_to_json(self):
        """센트로이드를 JSON 파일로 저장"""
        room_data = {}
        for i, centroid in enumerate(self.centroid_list):
            room_name = f"room{i+1}"
            room_data[room_name] = {"x": float(centroid[0]), "y": float(centroid[1])}
        
        with open(self.json_output, 'w') as json_file:
            json.dump(room_data, json_file, indent=4)
        
        print(f"센트로이드 데이터가 JSON 파일로 저장됨: {self.json_output}")
    
    def visualize_segmentation_steps(self):
        """세그먼테이션 단계별 시각화"""
        # 경계선 표시 이미지
        img_contours = self.create_contour_image()
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
        plt.title('SLIC Superpixel Segmentation (Map Interior Only)')
        plt.show()
        
        # 색상 입힌 세그먼트 이미지
        img_color = self.create_colored_segments()
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title('SLIC Superpixel Segmentation with Colored Regions')
        plt.show()
        
        # 세그먼테이션 결과 시각화 (폴리곤 포함)
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        for polygon in self.polygon_list_map:
            polygon = np.array(polygon)
            plt.plot(polygon[:, 0], polygon[:, 1], linewidth=2)
        plt.title('Segmented Map with Filtered Small Regions')
        plt.show()
        
        # 최종 이미지
        final_img = self.create_final_image()
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        plt.title('Final Segmented Map with Centroids')
        plt.show()
        
        return final_img
    
    def show_combined_result(self):
        """결과를 하나의 창에 결합하여 표시"""
        # 경계선 표시 이미지
        img_contours = self.create_contour_image()
        
        # 색상 입힌 세그먼트 이미지
        img_color = self.create_colored_segments()
        
        # 최종 이미지
        final_img = self.create_final_image()
        
        # 결합 이미지 생성
        top_row = np.hstack((self.img, img_contours))
        bottom_row = np.hstack((img_color, final_img))
        combined_img = np.vstack((top_row, bottom_row))
        
        # OpenCV 창에 표시
        cv2.imshow("Map Segmentation Result", cv2.resize(combined_img, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return combined_img
    
    def run(self, visualize=True):
        """전체 프로세스 실행"""
        # 전처리: 흰색 영역(맵 내부) 추출
        self.preprocess_image()
        
        # SLIC 알고리즘 적용
        self.apply_slic()
        
        # 폴리곤 추출
        self.extract_polygons()
        
        # 최종 이미지 생성
        final_img = self.create_final_image()
        
        # 결과 이미지 저장
        self.save_result(final_img)
        
        # 센트로이드를 JSON 파일로 저장
        self.save_centroids_to_json()
        
        # 시각화 (선택적)
        if visualize:
            self.show_combined_result()
        
        return final_img


def main():
    parser = argparse.ArgumentParser(description="Map Interior Room Segmentation")
    parser.add_argument("--img_path", type=str, required=True, help="입력 이미지 파일 경로 (PGM, PNG 등)")
    parser.add_argument("--n_segments", type=int, default=35, help="세그먼트 개수")
    parser.add_argument("--min_area", type=int, default=500, help="최소 세그먼트 영역 크기 (픽셀)")
    parser.add_argument("--output", type=str, default="./data/output_map.png", help="출력 이미지 파일명")
    parser.add_argument("--json_output", type=str, default="./data/centroids.json", help="센트로이드 JSON 출력 파일명")
    parser.add_argument("--slic_iter", type=int, default=100, help="SLIC 알고리즘 반복 횟수")
    parser.add_argument("--ruler", type=int, default=100, help="SLIC 알고리즘 ruler 파라미터")
    parser.add_argument("--no_visualize", action="store_true", help="시각화 비활성화")
    
    args = parser.parse_args()
    
    segmentation = RoomSegmentation(
        img_path=args.img_path,
        n_segments=args.n_segments,
        min_area=args.min_area,
        output_file=args.output,
        json_output=args.json_output,
        slic_iter=args.slic_iter,
        ruler=args.ruler
    )
    
    segmentation.run(visualize=not args.no_visualize)


if __name__ == "__main__":
    main()