const awsSDK = require('aws-sdk');
const fs = require('fs');
const mime = require('mime-types');
const { isImage } = require('./fileExtension');
const axios = require('axios');
const Record = require('../models/Record');
const sharp = require('sharp');
sharp.cache(false);

exports.uploadS3 = async function (req, res, next) {
	if (req.files) {
		try {
			let isValid = true;

			await req.files.forEach(function (file) {
				if (!isImage(file.path)) {
					isValid = false;
				}
			});

			if (!isValid) {
				await req.files.forEach(function (file) {
					fs.unlinkSync(file.path);
				});
				return next({
					message: 'One of the files is not a valid image file',
				});
			}

			let URLs = [];
			let cannyURLs = [];
			let laplacianURLs = [];
			let sobelXURLs = [];
			let sobelYURLs = [];
			let otsuURLs = [];
			let cannyArea = [];
			let cannyPerimeter = [];
			let laplacianArea = [];
			let laplacianPerimeter = [];
			let sobelXArea = [];
			let sobelXPerimeter = [];
			let sobelYArea = [];
			let sobelYPerimeter = [];
			let otsuArea = [];
			let otsuPerimeter = [];

			req.files.forEach(async (file) => {
				await resizeFile(file.path);

				await uploadFile(file.filename, file.path);
				URLs.push(process.env.CLOUDFRONT_URL + file.filename);
				let canny = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/canny1',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				cannyURLs.push(canny.data.url);
				cannyArea.push(canny.data.area);
				cannyPerimeter.push(canny.data.perimeter);
				canny = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/canny2',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				cannyURLs.push(canny.data.url);
				cannyArea.push(canny.data.area);
				cannyPerimeter.push(canny.data.perimeter);
				canny = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/canny3',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				cannyURLs.push(canny.data.url);
				cannyArea.push(canny.data.area);
				cannyPerimeter.push(canny.data.perimeter);
				let laplacian = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/laplacian',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				laplacianURLs.push(laplacian.data.url);
				laplacianArea.push(laplacian.data.area);
				laplacianPerimeter.push(laplacian.data.perimeter);
				let sobelX = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/sobelx',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				sobelXURLs.push(sobelX.data.url);
				sobelXArea.push(sobelX.data.area);
				sobelXPerimeter.push(sobelX.data.perimeter);
				let sobelY = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/sobely',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				sobelYURLs.push(sobelY.data.url);
				sobelYArea.push(sobelY.data.area);
				sobelYPerimeter.push(sobelY.data.perimeter);

				let otsu = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/otsu',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				otsuURLs.push(otsu.data.url);
				otsuArea.push(otsu.data.area);
				otsuPerimeter.push(otsu.data.perimeter);

				fs.unlinkSync(file.path);
				if (
					cannyURLs.length === 3 &&
					laplacianURLs.length === req.files.length &&
					sobelXURLs.length === req.files.length &&
					sobelYURLs.length === req.files.length &&
					otsuURLs.length === req.files.length
				) {
					const record = await Record.create({
						imageUrl: URLs[0],
						cannyUrl1: cannyURLs[0],
						cannyUrl2: cannyURLs[1],
						cannyUrl3: cannyURLs[2],
						sobelXUrl: sobelXURLs[0],
						sobelYUrl: sobelYURLs[0],
						laplacianUrl: laplacianURLs[0],
						otsuUrl: otsuURLs[0],
						cannyArea1: cannyArea[0],
						cannyPerimeter1: cannyPerimeter[0],
						cannyArea2: cannyArea[1],
						cannyPerimeter2: cannyPerimeter[1],
						cannyArea3: cannyArea[2],
						cannyPerimeter3: cannyPerimeter[2],
						laplacianArea: laplacianArea[0],
						laplacianPerimeter: laplacianPerimeter[0],
						sobelXArea: sobelXArea[0],
						sobelXPerimeter: sobelXPerimeter[0],
						sobelYArea: sobelYArea[0],
						sobelYPerimeter: sobelYPerimeter[0],
						otsuArea: otsuArea[0],
						otsuPerimeter: otsuPerimeter[0],
					});
					console.log(record);
					return res.status(200).json(record);
				}
			});
		} catch (e) {
			req.files.forEach((file) => fs.unlinkSync(file.path));
			return next(e);
		}
	} else {
		return next('File not upladed');
	}
};

// exports.uploadS3 = async function (req, res, next) {
//     try {
//         await uploadFile(req.file.filename, req.file.path);
//         let urlResponse = process.env.CLOUDFRONT_URL + req.file.filename;
//         fs.unlinkSync(req.file.path);
//         return res.status(200).json(urlResponse);
//     } catch (e) {
//         return next(e);
//     }
// };

async function uploadFile(filename, fileDirectoryPath) {
	awsSDK.config.update({
		accessKeyId: process.env.S3_ACCESS_KEY_ID,
		secretAccessKey: process.env.S3_SECRET_ACCESS_KEY,
	});
	const s3 = new awsSDK.S3();

	return new Promise(function (resolve, reject) {
		fs.readFile(fileDirectoryPath.toString(), function (err, data) {
			if (err) {
				reject(err);
			}
			const conType = mime.lookup(fileDirectoryPath);
			s3.putObject(
				{
					Bucket: '' + process.env.S3_BUCKET_NAME,
					Key: filename,
					Body: data,
					ContentType: conType,
					ACL: 'public-read',
				},
				function (err, data) {
					if (err) reject(err);
					resolve('successfully uploaded');
				}
			);
		});
	});
}

async function resizeFile(path) {
	let buffer = await sharp(path)
		.resize(200, 200, {
			fit: sharp.fit.inside,
			withoutEnlargement: true,
		})
		.toBuffer();
	return sharp(buffer).toFile(path);
}
