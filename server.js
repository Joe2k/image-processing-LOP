const express = require('express');
const app = express();
const passport = require('passport');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const cookieSession = require('cookie-session');
const errorHandler = require('./utils/error');
const path = require('path');
require('dotenv/config');

//MIDDLEWARES
app.use(cors());
app.use(
	bodyParser.urlencoded({
		extended: true,
	})
);
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'client', 'build')));

//IMPORT SCHEMAS AND CONNECT TO DB
require('./models/User.js');
mongoose.set('useCreateIndex', true);
mongoose.set('useUnifiedTopology', true);
mongoose.set('useFindAndModify', false);
mongoose.connect(
	process.env.MONGODB_URI,
	{
		useNewUrlParser: true,
		useUnifiedTopology: true,
	},
	() => console.log('Connected to MongoDB')
);

app.use(
	cookieSession({
		maxAge: 30 * 24 * 60 * 60 * 1000,
		keys: ['somesecretsauce'],
	})
);

//  PASSPORT AUTH
app.use(passport.initialize());
app.use(passport.session());
require('./utils/passport.js');
require('./routes/api/auth.js')(app);
const uploadRoute = require('./routes/api/upload.js');
app.use('/api', uploadRoute);
const userRoute = require('./routes/api/user.js');
app.use('/user', userRoute);
const recordRoute = require('./routes/api/record.js');
app.use('/record', recordRoute);

// ROUTES
app.get('/api', (req, res) => {
	res.send({
		message: 'Welcome to the project API',
	});
});

app.get('*', (req, res) => {
	res.sendFile(path.join(__dirname, 'client', 'build', 'index.html'));
});

app.use(errorHandler);

PORT = process.env.PORT || 4050;
app.listen(PORT, function () {
	console.log('Listening on port', PORT);
	console.log(`Open here http://localhost:${PORT}`);
});

module.exports = app;
