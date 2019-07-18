import { Component, OnInit } from '@angular/core';
import * as CanvasJS from '../../assets/canvasjs.min.js'
import {generateClassesFrequency} from '../../assets/sharedData'


@Component({
  selector: 'app-stats',
  templateUrl: './stats.component.html',
  styleUrls: ['./stats.component.scss']
})
export class StatsComponent implements OnInit {
localeData: any;
cadData: any;
aparallelData: any;

  constructor() {}
  ngOnInit() {
    this.initializeGraphData();
    this.populateLocaleData();
    this.populateGraph('Locale Graph',this.localeData);
  }
  populateLocaleData(){
    this.populateGraph('Locale Graph',this.localeData);
   }
  populateAparallelData(){
    this.populateGraph('Apparel',this.aparallelData);
   }
  populateCadData(){
    this.populateGraph('Camera Angle Graph',this.cadData);
   }
  initializeGraphData(){
    this.localeData = generateClassesFrequency('locale');
    this.cadData = generateClassesFrequency('cad')
    this.aparallelData = generateClassesFrequency('aparallel')
  }
  populateGraph(graphName:string,data){
    this.drawBarChart(graphName,data);
    this.drawPieChart(graphName,data);
  }
  drawBarChart(title,dataPlots){
    let chart = new CanvasJS.Chart("barChartContainer", {
      animationEnabled: true,
      exportEnabled: true,
      title: {
        text: title
      },
      data: [{
        type: "column",
        dataPoints: dataPlots
      }]
    });
    chart.render();
  }
  drawPieChart(title,dataPlots){
    let chart = new CanvasJS.Chart("pieChartContainer", {
      animationEnabled: true,
      exportEnabled: true,
      title: {
        text:title
      },
      data: [{
        type: "pie",
        dataPoints: dataPlots
      }]
    });
    chart.render();
  }
}
