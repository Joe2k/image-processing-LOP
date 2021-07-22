const Record = require("../models/Record");
const axios = require("axios");

exports.postRecord = async function (req, res, next) {
  //console.log(req.body.points);
  let numberOfPoints = 0;
  req.body.points.lines.forEach((line) => {
    numberOfPoints += line.points.length;
  });
  let resp = await axios({
    method: "post",
    url: "http://127.0.0.1:4000/process",
    data: req.body.points,
  });

  const record = await Record.findByIdAndUpdate(
    req.body._id,
    {
      points: req.body.points,
      numberOfPoints,
      originalPerimeter: resp.data.perimeter,
      originalArea: resp.data.area,
    },
    { new: true }
  );

  console.log(resp.data);
  return res.json(record);
};

exports.getRecord = async function (req, res, next) {
  const record = await Record.findById(req.params.id);
  return res.json(record);
};
