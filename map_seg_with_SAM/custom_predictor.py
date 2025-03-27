import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import os
import torch
import json
import matplotlib
matplotlib.use('TkAgg')  # GUI 백엔드 설정

class ClickBasedSegmenter:
    def __init__(self, image, predictor):
        self.image = image
        self.predictor = predictor
        self.masks = []
        self.current_room = 1
        self.centroids = {}
        self.colored_mask = np.zeros_like(image)
        self.combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        self.room_colors = {}
        self.current_mask = None
        self.click_stack = [[], []]  # [coords], [modes]
        self.room_masks = {}  # 각 방별 마스크 저장
        
        # 그림 초기화
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.imshow(self.image)
        self.ax.set_title('클릭으로 영역 생성: 좌클릭=전경, 우클릭=배경, Enter=방 생성, C=취소, ESC=종료')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # 안내 텍스트 추가
        self.help_text = self.ax.text(10, 20, 
                                      "좌클릭=객체 선택, 우클릭=배경 선택, Enter=방 생성, C=취소, N=다음 객체, ESC=종료",
                                      color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        # 현재 작업 중인 마스크 표시를 위한 이미지 레이어
        self.mask_layer = None
        
        # 현재 방 번호 표시
        self.room_text = self.ax.text(10, 50, f"현재 방 번호: {self.current_room}", 
                                      color='white', bbox=dict(facecolor='black', alpha=0.7))
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
            
        # 클릭한 좌표 저장
        x, y = int(event.xdata), int(event.ydata)
        
        # 이미 존재하는 방 영역 클릭 확인
        if self.combined_mask[y, x]:
            print(f"경고: 이미 다른 방이 있는 영역입니다. 빈 공간을 선택해주세요.")
            return
        
        # 버튼에 따라 전경(1) 또는 배경(0) 선택
        if event.button == 1:  # 좌클릭 = 전경
            mode = 1
            marker = 'ro'
        elif event.button == 3:  # 우클릭 = 배경
            mode = 0
            marker = 'bx'
        else:
            return
        
        self.click_stack[0].append([x, y])
        self.click_stack[1].append(mode)
        
        # 클릭 위치 표시
        self.ax.plot(x, y, marker, markersize=5)
        
        # SAM에 좌표 입력하여 마스크 생성
        self.update_mask()
        
        self.fig.canvas.draw()
        print(f"포인트 추가: ({x}, {y}), 모드: {'전경' if mode == 1 else '배경'}")
    
    def update_mask(self):
        if not self.click_stack[0]:
            return
            
        # SAM 모델에 좌표 입력
        input_points = np.array(self.click_stack[0])
        input_labels = np.array(self.click_stack[1])
        
        # 마스크 예측
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # 최고 점수의 마스크 선택
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # 겹치는 영역 제거 (기존 마스크 우선)
        filtered_mask = np.logical_and(best_mask, ~self.combined_mask)
        
        # 유효성 확인 - 마스크가 너무 작으면 제거된 후의 마스크 사용하지 않고 원본 사용
        if np.sum(filtered_mask) < 100 and np.sum(best_mask) > 500:
            print("경고: 기존 방과 겹치는 부분이 많습니다!")
            # filtered_mask = best_mask  # 제거하지 않고 원본 사용
        
        # 기존 마스크 레이어 제거
        if self.mask_layer is not None:
            self.mask_layer.remove()
            
        # 새 마스크 시각화 (반투명)
        color_overlay = np.zeros_like(self.image, dtype=np.uint8)
        color_overlay[filtered_mask] = [0, 255, 0]  # 녹색으로 표시
        
        # 겹치는 영역 빨간색으로 표시
        overlap_mask = np.logical_and(best_mask, self.combined_mask)
        if np.any(overlap_mask):
            color_overlay[overlap_mask] = [255, 0, 0]  # 빨간색으로 표시 (겹침 경고)
        
        self.mask_layer = self.ax.imshow(color_overlay, alpha=0.4)
        
        # 현재 마스크 저장
        self.current_mask = filtered_mask
        self.full_mask = best_mask  # 원본 마스크도 저장
    
    def on_key(self, event):
        if event.key == 'enter':
            if self.current_mask is not None:
                self.confirm_room()
            
        elif event.key == 'c':  # 취소
            self.cancel_current()
            
        elif event.key == 'n':  # 다음 객체
            self.next_object()
            
        elif event.key == 'escape':
            plt.close(self.fig)
    
    def cancel_current(self):
        # 현재 작업 취소
        self.click_stack = [[], []]
        
        # 마스크 레이어 제거
        if self.mask_layer is not None:
            self.mask_layer.remove()
            self.mask_layer = None
            
        self.current_mask = None
        self.fig.canvas.draw()
        print("현재 작업이 취소되었습니다.")
    
    def next_object(self):
        # 다음 객체로 넘어가기
        if self.current_mask is not None:
            self.confirm_room()
            
        self.current_room += 1
        
        # 방 번호 표시 업데이트
        self.room_text.set_text(f"현재 방 번호: {self.current_room}")
        self.fig.canvas.draw()
        
        print(f"다음 객체 ({self.current_room})를 선택하세요")
    
    def confirm_room(self):
        # 유효성 검사
        if self.current_mask is None or np.sum(self.current_mask) < 100:
            print("유효한 마스크가 없습니다. 다시 선택해주세요.")
            return
        
        # 마스크 저장
        self.masks.append(self.current_mask)
        self.room_masks[self.current_room] = self.current_mask  # 방 번호별 마스크 저장
        self.combined_mask = np.logical_or(self.combined_mask, self.current_mask)
        
        # 마스크 색상 생성 및 적용
        color = np.random.randint(0, 255, size=3)
        self.colored_mask[self.current_mask] = color
        self.room_colors[self.current_room] = color
        
        # 중심점 계산
        y_indices, x_indices = np.where(self.current_mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            centroid_y = int(np.mean(y_indices))
            centroid_x = int(np.mean(x_indices))
            
            # 중심점 저장
            self.centroids[f"room{self.current_room}"] = {"x": centroid_x, "y": centroid_y}
        else:
            print("경고: 마스크가 비어있습니다!")
            return
        
        # 마스크 레이어 제거
        if self.mask_layer is not None:
            self.mask_layer.remove()
            self.mask_layer = None
        
        # 완성된 마스크 표시
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.imshow(self.colored_mask, alpha=0.5)
        
        # 중심점 표시
        for room_idx, centroid in self.centroids.items():
            room_num = int(room_idx.replace('room', ''))
            self.ax.plot(centroid['x'], centroid['y'], 'bo', markersize=8)
            self.ax.text(centroid['x'], centroid['y'], f'Room {room_num}',
                    color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
        
        # 안내 텍스트 다시 추가
        self.help_text = self.ax.text(10, 20, 
                                      "좌클릭=객체 선택, 우클릭=배경 선택, Enter=방 생성, C=취소, N=다음 객체, ESC=종료", 
                                      color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        # 현재 방 번호 표시 업데이트
        self.room_text = self.ax.text(10, 50, f"현재 방 번호: {self.current_room}", 
                                      color='white', bbox=dict(facecolor='black', alpha=0.7))
        
        self.fig.canvas.draw()
        print(f"방 {self.current_room} 생성 완료")
        
        # 다음 마스크를 위한 설정 초기화
        self.click_stack = [[], []]
        self.current_mask = None
        
        # 자동으로 다음 방 번호로 이동
        self.current_room += 1
        self.room_text.set_text(f"현재 방 번호: {self.current_room}")
        
    def save_results(self, output_path):
        # 기본 파일 이름과 확장자 분리
        base_name, ext = os.path.splitext(output_path)
        
        # 최종 결과 저장 (세그멘테이션 이미지)
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        plt.imshow(self.colored_mask, alpha=0.5)
        
        # 방 번호와 중심점 표시
        for room_idx, centroid in self.centroids.items():
            room_num = int(room_idx.replace('room', ''))
            plt.plot(centroid['x'], centroid['y'], 'bo', markersize=8)
            plt.text(centroid['x'], centroid['y'], f'Room {room_num}',
                    color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
                    
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        
        # 개별 방 마스크 이미지 저장
        output_dir = os.path.dirname(output_path)
        if not output_dir:
            output_dir = "."
        
        mask_dir = os.path.join(output_dir, "room_masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        # 각 방별 개별 마스크 저장
        for room_num, mask in self.room_masks.items():
            room_img = np.zeros_like(self.image)
            room_img[mask] = self.room_colors[room_num]
            mask_path = os.path.join(mask_dir, f"room_{room_num}{ext}")
            
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            plt.imshow(room_img, alpha=0.5)
            plt.axis('off')
            plt.savefig(mask_path)
            plt.close()
        
        # 중심점 JSON 저장
        json_path = f"{base_name}_centroids.json"
        with open(json_path, 'w') as f:
            json.dump(self.centroids, f, indent=4)
            
        print(f"세그멘테이션 결과가 {output_path}에 저장되었습니다.")
        print(f"개별 방 마스크가 {mask_dir} 폴더에 저장되었습니다.")
        print(f"방 중심점이 {json_path}에 저장되었습니다.")

def segment_map_with_sam(map_path, output_path):
    # 1. 모델 로드
    sam_checkpoint = "/home/yeseul/Desktop/cnrlab/sam_vit_l_0b3195.pth"
    model_type = "vit_l"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # 2. 맵 이미지 로드
    image = cv2.imread(map_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. 이미지 임베딩 계산
    predictor.set_image(image)
    
    # 4. 클릭 기반 세그멘테이션 시작
    segmenter = ClickBasedSegmenter(image, predictor)
    plt.show()  # 인터랙티브 창 표시
    
    # 5. 결과 저장
    segmenter.save_results(output_path)
    
    return segmenter.colored_mask, segmenter.centroids

# 사용 방법 안내 출력
print("사용 방법:")
print("1. 좌클릭으로 객체(전경)를 선택하세요.")
print("2. 우클릭으로 배경을 선택하세요.")
print("3. Enter 키를 눌러 현재 마스크를 확정하세요. (자동으로 다음 방 번호로 넘어갑니다)")
print("4. C 키를 눌러 현재 작업을 취소할 수 있습니다.")
print("5. N 키를 눌러 다음 객체로 수동으로 넘어갈 수 있습니다.")
print("6. 모든 방 지정이 끝나면 ESC 키를 눌러 종료하세요.")
print("7. 이미 지정된 방 영역을 클릭하면 경고 메시지가 표시됩니다.")

# 실행 예시
map_path = "/home/yeseul/Documents/map_finish.png"
output_path = "./map_finish_seg2.png"
colored_mask, centroids = segment_map_with_sam(map_path, output_path)
