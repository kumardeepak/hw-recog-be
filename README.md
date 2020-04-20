# hw-recog-be
handwriting recognition REST backend

Avilable API routes

   '/tables_extractor/rois' (POST, OPTIONS) -> tables_rois_controllers.process_image>,
  
  '/tables_lines/detect' (POST, OPTIONS) -> tables_lines_controllers.process_image>,
  
   '/users/me' (GET, HEAD, OPTIONS) -> users.me>,
  
   '/api/info' (HEAD, GET, POST, OPTIONS) -> info_controllers.info>,
  
   '/static/<filename>' (GET, HEAD, OPTIONS) -> static>]
