from pathlib import Path
from aiohttp import web
import asyncio
import aiohttp_jinja2
import jinja2


if "__file__" in globals():
    BASE_DIR = Path(__file__).parent.parent
else:
    import os
    BASE_DIR = Path(os.getcwd()) / "server"

TEMPLATE_DIR = BASE_DIR / "foodie" / "templates"
STATIC_DIR = BASE_DIR / "foodie" / "static"
IMAGE_DIR = STATIC_DIR / "images"


################################################################################
#                                VIEWS                                         #
################################################################################

async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name
    return web.Response(text=text)


@aiohttp_jinja2.template('index.html')
async def index(request):
    return {"questions": ["how do you do?", "are you fine?"]}


async def upload_image(request):
    # data = await request.post()
    # img = data["img"]
    # filename = img.filename
    # mp3_file = data["img"].file
    # content = img.read()
    # with open(IMAGE_DIR / filename, "wb") as outf:
    #     outf.write(content)
    # return web.Response(body=content,
    #                     headers=MultiDict(
    #                         {'CONTENT-DISPOSITION': img}))
    reader = await request.multipart()
    field = await reader.next()
    # print(field.name)
    # assert field.name == "name"
    # name = await field.read(decode=True)
    # field = await reader.next()
    assert field.name == "img"
    filename = field.filename
    # You cannot rely on Content-Length if transfer is chunked.
    size = 0
    with open(IMAGE_DIR / filename, 'wb') as f:
        while True:
            chunk = await field.read_chunk()  # 8192 bytes by default.
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    return web.Response(text='{} sized of {} successfully stored'
                             ''.format(filename, size))



def setup_static_routes(app):
    app.router.add_static('/static/', path=STATIC_DIR, name='static')


################################################################################
#                                  SETUP                                       #
################################################################################

app = web.Application()
aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)))
setup_static_routes(app)
app.add_routes([web.get("/", index),
                web.get("/{name}", handle),
                web.post("/user/{user_id}/upload", upload_image)])


def start():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(asyncio.new_event_loop())
    web.run_app(app, port=80)


if __name__ == '__main__':
    start()
