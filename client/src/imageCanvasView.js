import './App.css';
import { useState, useEffect } from 'react';
import base64 from 'base-64';
import CanvasDraw from 'react-canvas-draw';
import Button from 'react-bootstrap/Button';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import axios from 'axios';

function App(props) {
	let saveableCanvas;
	let [url, setUrl] = useState(null);
	let [points, setPoints] = useState(null);
	const [perimeter, setPerimeter] = useState(null);
	const [area, setArea] = useState(null);
	// let [load, setLoad] = useState(false);
	useEffect(() => {
		setUrl(base64.decode(props.match.params.url));
		console.log(base64.decode(props.match.params.url));
		axios.get('/record/get/' + props.match.params.id).then((response) => {
			console.log(response);
			setPoints(JSON.stringify(response.data.points));
			setPerimeter(response.data.originalPerimeter);
			setArea(response.data.originalArea);
		});
	}, []);
	return (
		url && (
			<Container fluid>
				<Row>
					<Col className="justify-content-md-center">
						<h1 style={{ fontSize: '26px', marginTop: '20px' }}>
							Original Image
						</h1>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<CanvasDraw
							ref={(canvasDraw) => (saveableCanvas = canvasDraw)}
							imgSrc={url}
							brushRadius={0}
							hideGrid={true}
							brushColor="#fff"
							style={{ margin: '20px', display: 'inline-block' }}
							canvasHeight={200}
							canvasWidth={200}
							lazyRadius={0}
							disabled={true}
						></CanvasDraw>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<Button
							style={{ width: '250px', margin: '25px' }}
							onClick={() => {
								//console.log(points);
								saveableCanvas.loadSaveData(points, true);
							}}
						>
							Toggle drawings
						</Button>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<h6>Perimeter : {perimeter} pixels</h6>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<h6>Area : {area} pixels</h6>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<Button
							style={{ width: '250px', margin: '25px' }}
							onClick={() => {
								//console.log(points);
								window.ReactNativeWebView.postMessage(
									'go back'
								);
							}}
						>
							Click to go back to processed images
						</Button>
					</Col>
				</Row>
			</Container>
		)
	);
}

export default App;
