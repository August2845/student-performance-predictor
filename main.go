package main

import (
	"encoding/csv"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
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

// ---------------------- Математика ----------------------

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

// ---------------------- Модель kNN ----------------------

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

func prepareXY(data []Student) ([][]float64, []float64) {
	X := make([][]float64, len(data))
	y := make([]float64, len(data))
	for i, s := range data {
		X[i] = []float64{s.Attendance, s.Homework, s.TestScore}
		y[i] = s.FinalScore
	}
	return X, y
}

// ---------------------- Глобальные данные модели ----------------------

var (
	trainData []Student
	Xtrain    [][]float64
	ytrain    []float64
)

// ---------------------- Веб-интерфейс ----------------------

const htmlTemplate = `
<!DOCTYPE html>
<html lang="ru">
<head>
	<meta charset="UTF-8">
	<title>Прогноз успеваемости студентов</title>
	<style>
		body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; background: #f7f7f7; }
		h1 { text-align: center; color: #333; }
		form { display: flex; flex-direction: column; gap: 10px; }
		label { font-weight: bold; }
		input { padding: 8px; font-size: 14px; border-radius: 8px; border: 1px solid #aaa; }
		button { padding: 10px; border: none; background: #007bff; color: white; font-weight: bold; border-radius: 8px; cursor: pointer; }
		button:hover { background: #0056b3; }
		.result { background: white; padding: 15px; margin-top: 20px; border-radius: 10px; box-shadow: 0 0 8px rgba(0,0,0,0.1); }
	</style>
</head>
<body>
	<h1>🎓 Прогноз успеваемости студента</h1>
	<form action="/" method="POST">
		<label>Посещаемость (%)</label>
		<input type="number" step="0.1" name="attendance" required>
		<label>Домашние задания (%)</label>
		<input type="number" step="0.1" name="homework" required>
		<label>Тесты (%)</label>
		<input type="number" step="0.1" name="testscore" required>
		<button type="submit">Предсказать</button>
	</form>
	{{if .Show}}
	<div class="result">
		<h3>Результат:</h3>
		<p>📊 Прогнозируемый итоговый балл: <strong>{{printf "%.2f" .Prediction}}</strong></p>
	</div>
	{{end}}
</body>
</html>
`

func handler(w http.ResponseWriter, r *http.Request) {
	tmpl := template.Must(template.New("index").Parse(htmlTemplate))
	if r.Method == "GET" {
		tmpl.Execute(w, nil)
		return
	}

	// Обработка формы
	attendance, _ := strconv.ParseFloat(r.FormValue("attendance"), 64)
	homework, _ := strconv.ParseFloat(r.FormValue("homework"), 64)
	testScore, _ := strconv.ParseFloat(r.FormValue("testscore"), 64)

	Xtest := [][]float64{{attendance, homework, testScore}}
	pred := predictKNN(Xtrain, ytrain, Xtest, 5)[0]

	data := struct {
		Show       bool
		Prediction float64
	}{
		Show:       true,
		Prediction: pred,
	}

	tmpl.Execute(w, data)
}

// ---------------------- main ----------------------

func main() {
	data := generateData(200)
	saveToCSV(data, "students.csv")

	trainData = data
	Xtrain, ytrain = prepareXY(trainData)

	http.HandleFunc("/", handler)
	fmt.Println("🌐 Сервер запущен: http://localhost:8080")
	http.ListenAndServe(":8080", nil)
}
