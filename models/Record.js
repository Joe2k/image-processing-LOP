var mongoose = require('mongoose');
var Schema = mongoose.Schema;

// SingleRegion
// SingleWatershed
// SingleWatershedRegion
// Otsu

var RecordSchema = new Schema({
	imageUrl: {
		type: String,
	},
	singleRegion: [
		{
			url: {
				type: String,
			},
			perimeter: {
				type: Number,
			},
			area: {
				type: Number,
			},
		},
	],

	singleWatershedRegion: [
		{
			url: {
				type: String,
			},
			perimeter: {
				type: Number,
			},
			area: {
				type: Number,
			},
		},
	],
	singleWatershed: [
		{
			url: {
				type: String,
			},
			perimeter: {
				type: Number,
			},
			area: {
				type: Number,
			},
		},
	],
	otsu: [
		{
			url: {
				type: String,
			},
			perimeter: {
				type: Number,
			},
			area: {
				type: Number,
			},
		},
	],
	points: {
		type: Object,
	},
	numberOfPoints: {
		type: Number,
	},
	originalArea: {
		type: Number,
	},
	originalPerimeter: {
		type: Number,
	},
});

module.exports = Record = mongoose.model('records', RecordSchema);
