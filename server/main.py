#coding=utf-8
import os, sys, platform
sys.path.append('..')
import posixpath
import BaseHTTPServer
from SocketServer import ThreadingMixIn
import urllib, urllib2
import cgi
import shutil
import mimetypes
import re
import time
import  test
import  tensorflow as tf
import  cv2
from cStringIO import StringIO
sess=tf.Session()
model_root='../model/model.cpkt'
tensorA,tensorB=test.load_model(sess,model_root+'.meta',model_root)


def showTips():
	print ""
	print '----------------------------------------------------------------------->> '
	try:
		port = int(sys.argv[1])
	except Exception, e:
		print '-------->> Warning: Port is not given, will use deafult port: 8080 '
		print '-------->> if you want to use other port, please execute: '
		print '-------->> python SimpleHTTPServerWithUpload.py port '
		print "-------->> port is a integer and it's range: 1024 < port < 65535 "
		port = 8011

	if not 1024 < port < 65535:  port = 8080
	# serveraddr = ('', port)
	print '-------->> Now, listening at port ' + str(port) + ' ...'
	osType = platform.system()
	if osType == "Linux":
		print '-------->> You can visit the URL:     http://192.168.11.105:' +str(port)#自己电脑的IP地址
	else:
		print '-------->> You can visit the URL:     http://127.0.0.1:' +str(port)
	print '----------------------------------------------------------------------->> '
	print ""
	return ('', port)

serveraddr = showTips()


class SimpleHTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):




	def do_GET(self):
		"""Serve a GET request."""
		# print "....................", threading.currentThread().getName()
		f = self.send_head()
		if f:
			self.copyfile(f, self.wfile)
			f.close()

	def do_HEAD(self):
		"""Serve a HEAD request."""
		f = self.send_head()
		if f:
			f.close()

	def do_POST(self):

		r,filepathname,info = self.deal_post_data()
		filename=os.path.basename(filepathname)
		shutil.move(filepathname,'upload/'+filename)
		filepathname='upload/'+filename
		origin_image=cv2.imread(filepathname)
		npA,npB=test.server(sess,origin_image,tensorA,tensorB,4)
		cv2.imwrite('npA.jpg',npA)
		cv2.imwrite('npB.jpg',npB)
		print r, info, "by: ", self.client_address
		f = StringIO()
		f.write('<meta charset="utf-8">')#使得网页可以显示中文
		f.write('<center>')
		f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
		f.write("<html>\n<title>Upload Result Page</title>\n")
		f.write("<body>\n<h1>光照智能处理（暂时只支持人脸光强不足、左右偏光、过爆）</h1>\n")


		f.write("<form ENCTYPE=\"multipart/form-data\" method=\"post\">")
		f.write("<input name=\"file\" type=\"file\" onChange=\"form.submit()\"/>")
		f.write("&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp")
		f.write("<input type=\"button\" value=\"首页\" onClick=\"location='/'\">")
		f.write("<hr>\n<ul>\n")







		if npA is None:
			f.write("<strong>未检测到人脸,无法进行光照处理\n</strong>")
		else:
			f.write('\n<h2>********************原始***************************************处理后**********************</h2>\n')
			f.write('<img src="npA.jpg" height="512" width="512"/>' )#插入图片,并设置图片大小
			f.write('*')
			f.write('<img src="npB.jpg" height="512" width="512"/>' )#插入图片,并设置图片大小


		f.write("<br><a href=\"%s\">back</a>" % self.headers['referer'])
		f.write("<hr><small>Powered By: bones7456, check new version at ")
		f.write("<a href=\"http://li2z.cn/?s=SimpleHTTPServerWithUpload\">")
		f.write("here</a>.</small></body>\n</html>\n")
		length = f.tell()
		f.seek(0)
		self.send_response(10)
		self.send_header("Content-type", "text/html")
		self.send_header("Content-Length", str(length))
		self.end_headers()

		if f:
			self.copyfile(f, self.wfile)
			f.close()

	#上传结束后,可以进行相关处理
	def deal_post_data(self):


		boundary = self.headers.plisttext.split("=")[1]
		remainbytes = int(self.headers['content-length'])

		line = self.rfile.readline()


		remainbytes -= len(line)

		if not boundary in line:
			return (False,False, "Content NOT begin with boundary")
		line = self.rfile.readline()
		remainbytes -= len(line)
		fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line)#fn返回:['被选中的文件名']

		if not fn:
			return (False,False, "Can't find out file name...")

		path = self.translate_path(self.path)#self.path默认为'/',通过该函数可以获取当前代码目录路径

		osType = platform.system()
		try:
			if osType == "Linux":
				fn = os.path.join(path, fn[0].decode('gbk').encode('utf-8'))
			else:
				fn = os.path.join(path, fn[0])
		except Exception, e:
			return (False, "文件名请不要用中文，或者使用IE上传中文名的文件。")
		#把fn=path+fn之后,就是保存到当前目录路径+文件名
		print 'newfn',fn
		'''while os.path.exists(fn):
			fn += "_"'''
		line = self.rfile.readline()
		remainbytes -= len(line)
		line = self.rfile.readline()
		remainbytes -= len(line)
		try:
			out = open(fn, 'wb')
		except IOError:
			return (False, "Can't create file to write, do you have permission to write?")

		preline = self.rfile.readline()
		remainbytes -= len(preline)
		while remainbytes > 0:
			line = self.rfile.readline()
			remainbytes -= len(line)
			if boundary in line:
				preline = preline[0:-1]
				if preline.endswith('\r'):
					preline = preline[0:-1]
				out.write(preline)
				out.close()

				return (True,fn,"File '%s' upload success!" % fn)
			else:
				out.write(preline)
				preline = line
		return (False, "Unexpect Ends of data.")

	def send_head(self):

		path = self.translate_path(self.path)

		f = None
		if os.path.isdir(path):
			if not self.path.endswith('/'):
				# redirect browser - doing basically what apache does
				self.send_response(301)
				self.send_header("Location", self.path + "/")
				self.end_headers()
				return None
			for index in "index.html", "index.htm":
				index = os.path.join(path, index)
				if os.path.exists(index):
					path = index
					break
			else:
				return self.list_directory(path)
		ctype = self.guess_type(path)
		try:
			# Always read in binary mode. Opening files in text mode may cause
			# newline translations, making the actual size of the content
			# transmitted *less* than the content-length!
			f = open(path, 'rb')
		except IOError:
			self.send_error(404, "File not found")
			return None
		self.send_response(10)
		self.send_header("Content-type", ctype)
		fs = os.fstat(f.fileno())
		self.send_header("Content-Length", str(fs[6]))
		self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
		self.end_headers()
		return f

	def list_directory(self, path):
		f = StringIO()
		displaypath = cgi.escape(urllib.unquote(self.path))
		f.write('<meta charset="utf-8">')
		f.write('<center>')
		f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
		f.write("<html>\n<title>Directory listing for %s</title>\n" % displaypath)
		f.write("<body>\n<h1>种族识别 %s<h1>\n" % displaypath)#字体大小为h1
		f.write("<hr>\n")

		'''f.write("<form method=\"post\" action=\"upload.php\" enctype=\"multipart/form-data\">")
  		f.write("<input name=\'uploads[]\' type=\"file\" multiple  onChange=\"form.submit() >")
  		f.write("<input type=\"submit\" value=\"Send\">")
		f.write("</form>")'''

		f.write("<form ENCTYPE=\"multipart/form-data\" method=\"post\">")
		f.write("<input name=\"file\" value=\"首页\" type=\"file\" onChange=\"form.submit()\" >")
		f.write("&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp")
		f.write("<input type=\"button\" value=\"首页\"  onClick=\"location='/'\">")
		f.write("<hr>\n<ul>\n")
		length = f.tell()
		f.seek(0)
		self.send_response(10)
		#self.send_header('Content-type','image/jpg')
		self.send_header("Content-type", "text/html")
		self.send_header("Content-Length", str(length))
		self.end_headers()
		return f

	def translate_path(self, path):

		# abandon query parameters
		path = path.split('?',1)[0]
		path = path.split('#',1)[0]
		path = posixpath.normpath(urllib.unquote(path))
		words = path.split('/')
		words = filter(None, words)
		path = os.getcwd()#获得当前路径
		for word in words:
			drive, word = os.path.splitdrive(word)
			head, word = os.path.split(word)
			if word in (os.curdir, os.pardir): continue
			path = os.path.join(path, word)
		return path

	def copyfile(self, source, outputfile):

		shutil.copyfileobj(source, outputfile)

	def guess_type(self, path):

		base, ext = posixpath.splitext(path)
		if ext in self.extensions_map:
			return self.extensions_map[ext]
		ext = ext.lower()
		if ext in self.extensions_map:
			return self.extensions_map[ext]
		else:
			return self.extensions_map['']

	if not mimetypes.inited:
		mimetypes.init() # try to read system mime.types
	extensions_map = mimetypes.types_map.copy()
	extensions_map.update({
		'': 'application/octet-stream', # Default
		'.py': 'text/plain',
		'.c': 'text/plain',
		'.h': 'text/plain',
		})

class ThreadingServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
	pass





srvr = ThreadingServer(serveraddr, SimpleHTTPRequestHandler)

srvr.serve_forever()
#sess.close()
