import numpy as np
import cv2
import argparse
import json 

class RoomSegmentation:
    def __init__(self, img_path, n_segments=80, min_area=500, output_file="output_map.png", json_output="centroids.json"):
        self.img_path = img_path
        self.n_segments = n_segments
        self.min_area = min_area
        self.output_file = output_file
        self.json_output = json_output  
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"cannot find image file: {img_path}")
        
        # variables for saving results
        self.labels = None
        self.mask = None
        self.polygon_list_map = []
        self.centroid_list = []
    
    def preprocess_image(self):
        """preprocess image"""
        self.img[self.img >= 250] = 0
        image_bit = cv2.bitwise_not(self.img)
        kernel = np.ones((3, 3), np.uint8)
        open_img = cv2.morphologyEx(image_bit, cv2.MORPH_OPEN, kernel, iterations=1)
        self.img = open_img
        return self.img
    
    def apply_slic(self):
        """apply SLIC algorithm"""
        height, width = self.img.shape[:2]
        region_size = int(np.sqrt((width * height) / self.n_segments))
        
        # create SLIC and calculate superpixels
        slic = cv2.ximgproc.createSuperpixelSLIC(self.img, region_size=region_size, ruler=20)
        slic.iterate(10)
        
        # get superpixel labels and contour mask
        self.labels = slic.getLabels()
        self.mask = slic.getLabelContourMask()
        
        return self.labels, self.mask
    
    def create_contour_image(self):
        """create image with contours"""
        img_with_contours = self.img.copy()
        img_with_contours[self.mask == 255] = [0, 255, 0]
        return img_with_contours
    
    def create_colored_segments(self):
        """apply color to segments with white areas"""
        img_color = self.img.copy()
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        white_mask = gray_img == 255
        
        unique_segments = np.unique(self.labels)
        
        for seg_id in unique_segments:
            mask_seg = (self.labels == seg_id) & white_mask
            if np.any(mask_seg):
                color = np.random.randint(0, 255, size=3).tolist()
                img_color[mask_seg] = color
        
        img_color[self.mask == 255] = [0, 0, 255]
        return img_color
    
    def extract_polygons(self):
        """extract polygons from segments"""
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        white_mask = gray_img == 255
        unique_segments = np.unique(self.labels)
        
        self.polygon_list_map = []
        self.centroid_list = []
        
        for seg_id in unique_segments:
            mask_seg = (self.labels == seg_id) & white_mask
            if np.any(mask_seg):
                mask_seg_uint8 = mask_seg.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_seg_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_area:
                        continue  # ignore small segments
                    
                    contour_map_coords = []
                    for point in contour:
                        map_x, map_y = point[0][0], point[0][1]
                        contour_map_coords.append([map_x, map_y])
                    
                    self.polygon_list_map.append(contour_map_coords)
                    
                    # first try moment method to calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])
                        
                        # if centroid is outside the polygon, use distance transform method
                        if cv2.pointPolygonTest(contour, (centroid_x, centroid_y), False) < 0:
                            # use distance transform method
                            mask_contour = np.zeros_like(mask_seg_uint8)
                            cv2.drawContours(mask_contour, [contour], 0, 255, -1)
                            
                            # apply distance transform
                            dist_transform = cv2.distanceTransform(mask_contour, cv2.DIST_L2, 5)
                            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                            
                            # use the point with the maximum distance as centroid
                            centroid_x, centroid_y = max_loc
                            
                            # if still outside, find a random point inside the polygon
                            if cv2.pointPolygonTest(contour, (centroid_x, centroid_y), False) < 0:
                                # sample a point inside the polygon
                                x, y, w, h = cv2.boundingRect(contour)
                                found = False
                                for _ in range(100):  # try up to 100 times
                                    sample_x = np.random.randint(x, x + w)
                                    sample_y = np.random.randint(y, y + h)
                                    if cv2.pointPolygonTest(contour, (sample_x, sample_y), False) >= 0:
                                        centroid_x, centroid_y = sample_x, sample_y
                                        found = True
                                        break
                                
                                # if still not found, use the first point of the polygon
                                if not found:
                                    centroid_x, centroid_y = contour[0][0]
                    else:
                        # if moment is 0 (rare case) use the first point of the polygon
                        centroid_x, centroid_y = contour[0][0]
                    
                    self.centroid_list.append([centroid_x, centroid_y])
        
        return self.polygon_list_map, self.centroid_list
    
    def create_final_image(self):
        """create final image (show polygons and centroids)"""
        img_with_polygons = self.img.copy()
        
        for polygon in self.polygon_list_map:
            polygon = np.array(polygon, dtype=np.int32)
            color = np.random.randint(0, 255, size=(3,)).tolist()
            cv2.fillPoly(img_with_polygons, [polygon], color=color)
        
        # add centroids to image    
        for centroid in self.centroid_list:
            cv2.circle(img_with_polygons, (centroid[0], centroid[1]), 5, (255, 0, 0), -1)
        
        return img_with_polygons
    
    def save_result(self, img):
        """save result image"""
        cv2.imwrite(self.output_file, img)
        print(f"result image saved: {self.output_file}")
    
    def save_centroids_to_json(self):
        """save centroids to JSON file"""
        room_data = {}
        
        for i, centroid in enumerate(self.centroid_list):
            room_name = f"room{i+1}"
            room_data[room_name] = {"x": float(centroid[0]), "y": float(centroid[1])}
        
        with open(self.json_output, 'w') as json_file:
            json.dump(room_data, json_file, indent=4)
        
        print(f"centroids data saved to JSON file: {self.json_output}")
    
    def run(self):
        """run entire process"""
        # preprocess image
        self.preprocess_image()
        
        # apply SLIC algorithm
        self.apply_slic()
        
        # create step-by-step images
        img_contours = self.create_contour_image()
        img_colored = self.create_colored_segments()
        
        # extract polygons
        self.extract_polygons()
        
        # create final image
        final_img = self.create_final_image()
        
        # save result
        self.save_result(final_img)
        
        # save centroids to JSON
        self.save_centroids_to_json()
        
        # show step-by-step images in OpenCV window
        top_row = np.hstack((self.img, img_contours))  
        bottom_row = np.hstack((img_colored, final_img))  
        combined_img = np.vstack((top_row, bottom_row))  

        cv2.imshow("Map Segmentation Result", cv2.resize(combined_img, None, fx=0.5, fy=0.5))
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    

def main():
    parser = argparse.ArgumentParser(description="automatic room segmentation")
    parser.add_argument("--img_path", type=str, required=True, help="input PGM file path")
    parser.add_argument("--n_segments", type=int, default=80, help="number of segments")
    parser.add_argument("--min_area", type=int, default=500, help="minimum segment area")
    parser.add_argument("--output", type=str, default="output_map.png", help="output image file name")
    parser.add_argument("--json_output", type=str, default="centroids.json", help="centroids JSON output file name")
    
    args = parser.parse_args()
    
    segmentation = RoomSegmentation(
        img_path=args.img_path,
        n_segments=args.n_segments,
        min_area=args.min_area,
        output_file=args.output,
        json_output=args.json_output
    )
    
    segmentation.run()

if __name__ == "__main__":
    main()