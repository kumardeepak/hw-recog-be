from api import app
from db.conmgr_mongo import connectmongo
connectmongo()

#prints all avilable routes
print( app.url_map)

app.run(host='0.0.0.0', port=6000)
