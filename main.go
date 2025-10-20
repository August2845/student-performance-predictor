package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"time"
)

// ---------------------- Структуры ----------------------

type Student struct {
	ID           int
	Attendance   float64
	Homework     float64
	TestScore    float64
	FinalScore   float64
	PredictedVal float64
}

// ---------------------- Генерация данных ----------------------

func generateData(n int) []Student {
	rand.Seed(time.Now().UnixNano())
	data := make([]Student, n)
	for i := 0; i < n; i++ {
		attendance := rand.Float64()*40 + 60
		homework := rand.Float64()*50 + 50
		testScore := rand.Float64()*50 + 50
		finalScore := 0.4*attendance + 0.3*homework + 0.3*testScore + rand.NormFloat64()*5
		if finalScore > 100 {
			finalScore = 100
		}
		if finalScore < 0 {
			finalScore = 0
		}
		data[i] = Student{i + 1, attendance, homework, testScore, finalScore, 0}
	}
	return data
}

// ---------------------- CSV ----------------------

func saveToCSV(data []Student, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal("Ошибка создания CSV:", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"ID", "Attendance", "Homework", "TestScore", "FinalScore"})

	for _, s := range data {
		writer.Write([]string{
			fmt.Sprintf("%d", s.ID),
			fmt.Sprintf("%.2f", s.Attendance),
			fmt.Sprintf("%.2f", s.Homework),
			fmt.Sprintf("%.2f", s.TestScore),
			fmt.Sprintf("%.2f", s.FinalScore),
		})
	}
	fmt.Println("✅ Данные сохранены в", filename)
}

// ---------------------- Вспомогательные ----------------------

func euclidDist(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += (a[i] - b[i]) * (a[i] - b[i])
	}
	return math.Sqrt(sum)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---------------------- k-NN модель ----------------------

func predictKNN(Xtrain [][]float64, ytrain []float64, Xtest [][]float64, k int) []float64 {
	n := len(Xtest)
	res := make([]float64, n)
	for i := 0; i < n; i++ {
		type pair struct {
			dist float64
			val  float64
		}
		arr := make([]pair, len(Xtrain))
		for j := 0; j < len(Xtrain); j++ {
			d := euclidDist(Xtest[i], Xtrain[j])
			arr[j] = pair{dist: d, val: ytrain[j]}
		}
		sort.Slice(arr, func(a, b int) bool {
			return arr[a].dist < arr[b].dist
		})
		var sum float64
		kk := min(k, len(arr))
		for t := 0; t < kk; t++ {
			sum += arr[t].val
		}
		res[i] = sum / float64(kk)
	}
	return res
}

func splitData(data []Student, ratio float64) (train, test []Student) {
	n := len(data)
	trainSize := int(float64(n) * ratio)
	rand.Shuffle(n, func(i, j int) { data[i], data[j] = data[j], data[i] })
	return data[:trainSize], data[trainSize:]
}

func prepareXY(data []Student) ([][]float64, []float64) {
	X := make([][]float64, len(data))
	y := make([]float64, len(data))
	for i, s := range data {
		X[i] = []float64{s.Attendance, s.Homework, s.TestScore}
		y[i] = s.FinalScore
	}
	return X, y
}

func mse(yTrue, yPred []float64) float64 {
	var sum float64
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}
	return sum / float64(len(yTrue))
}

// ---------------------- Глобальные данные ----------------------

var (
	trainData []Student
	Xtrain    [][]float64
	ytrain    []float64
)

// ---------------------- HTTP сервер ----------------------

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		var input struct {
			Attendance float64 `json:"attendance"`
			Homework   float64 `json:"homework"`
			TestScore  float64 `json:"testscore"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, "Ошибка входных данных", 400)
			return
		}

		Xtest := [][]float64{{input.Attendance, input.Homework, input.TestScore}}
		yPred := predictKNN(Xtrain, ytrain, Xtest, 5)

		result := map[string]float64{"predicted": yPred[0]}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
		return
	}

	tmpl := template.Must(template.New("index").Parse(htmlPage))
	tmpl.Execute(w, nil)
}

const htmlPage = `
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Прогноз успеваемости студентов</title>
<style>
body { font-family: Arial; background: #f8f9fa; margin: 50px; }
.container { max-width: 400px; background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
input { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ccc; border-radius: 5px; }
button { background: #007bff; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer; width: 100%; }
button:hover { background: #0056b3; }
h2 { text-align: center; }
</style>
</head>
<body>
<div class="container">
<h2>📈 Прогноз итоговой оценки</h2>
<label>Посещаемость (%)</label>
<input id="attendance" type="number" value="90" min="0" max="100">
<label>Домашние задания (%)</label>
<input id="homework" type="number" value="85" min="0" max="100">
<label>Результат тестов (%)</label>
<input id="testscore" type="number" value="80" min="0" max="100">
<button onclick="predict()">Рассчитать прогноз</button>
<h3 id="result"></h3>
</div>
<script>
async function predict() {
  const data = {
    attendance: parseFloat(document.getElementById('attendance').value),
    homework: parseFloat(document.getElementById('homework').value),
    testscore: parseFloat(document.getElementById('testscore').value)
  };
  const res = await fetch('/', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  const json = await res.json();
  document.getElementById('result').innerText = "Прогноз: " + json.predicted.toFixed(2);
}
</script>
</body>
</html>
`

// ---------------------- MAIN ----------------------

func main() {
	// 1. Генерируем данные
	data := generateData(200)
	saveToCSV(data, "students.csv")

	// 2. Делим на обучающие/тестовые
	train, _ := splitData(data, 0.8)
	Xtrain, ytrain = prepareXY(train)

	// 3. Запуск веб-интерфейса
	fmt.Println("🌐 Открой в браузере: http://localhost:8080")
	http.HandleFunc("/", predictHandler)
	http.ListenAndServe(":8080", nil)
}
