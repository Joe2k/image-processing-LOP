import './App.css';
import { useState, useEffect } from 'react';
import base64 from 'base-64';
import CanvasDraw from 'react-canvas-draw';
import Button from 'react-bootstrap/Button';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

function App(props) {
	let saveableCanvas;
	let [url, setUrl] = useState(null);
	// let [load, setLoad] = useState(false);
	useEffect(() => {
		setUrl(base64.decode(props.match.params.url));
		console.log(base64.decode(props.match.params.url));
	}, []);
	return (
		url && (
			<Container fluid>
				<Row>
					<Col className="justify-content-md-center">
						<h1 style={{ fontSize: '26px', marginTop: '20px' }}>
							Draw around the boils/scars and click on submit
						</h1>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<CanvasDraw
							ref={(canvasDraw) => (saveableCanvas = canvasDraw)}
							imgSrc={url}
							brushRadius={2}
							hideGrid={true}
							brushColor="#fff"
							style={{ margin: '20px', display: 'inline-block' }}
							canvasHeight={200}
							canvasWidth={200}
							lazyRadius={0}
						></CanvasDraw>
					</Col>
				</Row>
				<Row className="justify-content-md-center">
					<Col>
						<Button
							style={{ margin: '5px' }}
							variant="warning"
							onClick={() => {
								saveableCanvas.undo();
							}}
						>
							Undo
						</Button>
						<Button
							style={{ margin: '5px' }}
							variant="danger"
							onClick={() => {
								saveableCanvas.clear();
							}}
						>
							Clear
						</Button>
					</Col>
				</Row>
				<Row>
					<Col className="justify-content-md-center">
						<Button
							style={{ width: '250px', margin: '25px' }}
							onClick={() => {
								console.log(saveableCanvas.getSaveData());
								window.ReactNativeWebView.postMessage(
									saveableCanvas.getSaveData()
								);
							}}
						>
							Submit
						</Button>
					</Col>
				</Row>
			</Container>
		)
	);
}

export default App;
