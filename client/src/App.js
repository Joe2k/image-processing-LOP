import React from "react";
import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";
import ImageCanvas from "./imageCanvas";
import ImageCanvasView from "./imageCanvasView";

function App() {
  return (
    <div className="App">
      <Router>
        <Switch>
          <Route
            path={`/imageCrop/:url`}
            render={(props) => <ImageCanvas {...props} />}
          ></Route>
          <Route
            path={`/imageView/:url/:id`}
            render={(props) => <ImageCanvasView {...props} />}
          ></Route>
        </Switch>
      </Router>
    </div>
  );
}

export default App;
