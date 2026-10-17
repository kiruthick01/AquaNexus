/**
 * About: data, method and the limits worth knowing before relying on a number.
 */

export default function About() {
  return (
    <>
      <h1>About</h1>

      <section className="card">
        <h2>Data</h2>
        <ul className="prose">
          <li>
            <strong>Labels:</strong> about 565,000 dissolved-oxygen measurements from 5,883 river
            stations, 環境省 公共用水域水質測定, fiscal years 2013-22. All real, none synthetic.
          </li>
          <li>
            <strong>Weather:</strong> NASA POWER daily air temperature, rain, solar radiation and
            wind (MERRA-2 and GPM IMERG), on a 0.5° grid.
          </li>
          <li>
            <strong>Land use:</strong> 国土数値情報 土地利用細分メッシュ (FY2016, 100 m): the share of
            paddy, farm, forest, urban and water within 1 and 5 km of each station.
          </li>
          <li>
            <strong>Terrain:</strong> station elevation from 国土地理院's national elevation model,
            plus relief among stations within 10 km.
          </li>
        </ul>
      </section>

      <section className="card">
        <h2>Method</h2>
        <p className="prose">
          LightGBM regression. Evaluation never lets a test station or a test year leak into
          training: stations are held out whole and later fiscal years are held out in time. Neighbour
          and history features use only readings from other visits. Intervals are split-conformal,
          calibrated on stations the model did not fit.
        </p>
      </section>

      <section className="card">
        <h2>Limits</h2>
        <ul className="prose">
          <li>Monthly samples: it predicts oxygen on sampling days, not continuously or by the hour.</li>
          <li>
            Low oxygen is hard. Only 2% of samples are below 5 mg/L; the intervals cover far fewer
            than 90% of those, and the alert model raises many false alarms.
          </li>
          <li>
            Summer is the hardest season, and weather-only predictions for a new station explain
            about half the variance.
          </li>
          <li>
            There is no water temperature or river flow in the monitoring files, so neither is a
            model input.
          </li>
          <li>Proof of concept. Not regulatory or safety advice.</li>
        </ul>
      </section>
    </>
  );
}
