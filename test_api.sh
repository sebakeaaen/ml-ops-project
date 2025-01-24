curl -X 'POST' \
  'http://127.0.0.1:8080/classify/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'data=@data/raw/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Kirmizi_Pistachio/kirmizi (2).jpg'

curl -X 'POST' \
  'http://127.0.0.1:8080/metrics/' \
  -H 'accept: application/json' \
