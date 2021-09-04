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
			// SingleRegion
			// SingleWatershed
			// SingleWatershedRegion
			// Otsu

			let URLs = [];
			let singleRegion = [];
			let singleWatershed = [];
			let singleWatershedRegion = [];
			let otsu = [];

			req.files.forEach(async (file) => {
				await resizeFile(file.path);

				await uploadFile(file.filename, file.path);
				URLs.push(process.env.CLOUDFRONT_URL + file.filename);

				// SingleRegion
				let singleRegion1 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singleregion1',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let singleRegion2 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singleregion2',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let singleRegion3 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singleregion3',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				singleRegionData1 = singleRegion1.data;
				singleRegionData2 = singleRegion2.data;
				singleRegionData3 = singleRegion3.data;
				singleRegion = [
					singleRegionData1,
					singleRegionData2,
					singleRegionData3,
				];
				// SingleWatershed
				let singleWatershed1 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singlewatershed1',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let singleWatershed2 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singlewatershed2',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let singleWatershed3 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singlewatershed3',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				singleWaterData1 = singleWatershed1.data;
				singleWaterData2 = singleWatershed2.data;
				singleWaterData3 = singleWatershed3.data;
				singleWatershed = [
					singleWaterData1,
					singleWaterData2,
					singleWaterData3,
				];

				// SingleWatershedRegion
				let singleWatershedRegion1 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singlewatershedregion1',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let singleWatershedRegion2 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singlewatershedregion2',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let singleWatershedRegion3 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/singlewatershedregion3',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				singleWaterRegionData1 = singleWatershedRegion1.data;
				singleWaterRegionData2 = singleWatershedRegion2.data;
				singleWaterRegionData3 = singleWatershedRegion3.data;
				singleWatershedRegion = [
					singleWaterRegionData1,
					singleWaterRegionData2,
					singleWaterRegionData3,
				];

				// Otsu

				let otsu1 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/otsu1',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let otsu2 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/otsu2',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				let otsu3 = await axios({
					method: 'post',
					url: 'http://127.0.0.1:4000/otsu3',
					data: {
						url: process.env.CLOUDFRONT_URL + file.filename, // This is the body part
						filename: file.filename,
					},
				});
				otsuData1 = otsu1.data;
				otsuData2 = otsu2.data;
				otsuData3 = otsu3.data;
				otsu = [otsuData1, otsuData2, otsuData3];

				fs.unlinkSync(file.path);
				if (
					singleRegion.length === 3 &&
					otsu.length === 3 &&
					singleWatershed.length === 3 &&
					singleWatershedRegion.length === 3
				) {
					const record = await Record.create({
						imageUrl: URLs[0],
						singleRegion,
						singleWatershedRegion,
						singleWatershed,
						otsu,
					});
					console.log(record);
					return res.status(200).json(record);
				} else {
					console.log(singleRegion.length);
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
