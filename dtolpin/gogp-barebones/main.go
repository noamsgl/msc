package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/csv"
	"flag"
	"fmt"
	"gonum.org/v1/gonum/stat"
	"io"
	"math"
	"math/rand"
	. "noamsgl/kernel/ad"
	"os"
	"strconv"
	"time"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A toy GP regressor. Invocation:
  %s [OPTIONS] < INPUT > OUTPUT
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
	flag.Float64Var(&TRAIN, "train", TRAIN, "fraction of training data")
	flag.Float64Var(&NOISE, "noise", NOISE, "noise scaling")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.IntVar(&NPLATEAU, "nplateau", NPLATEAU, "number of plateau iterations")
	flag.Float64Var(&EPS, "eps", EPS, "optimization precision, in log-odds")
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	rand.Seed(time.Now().UTC().UnixNano())
}

var (
	TRAIN    = 0.
	NOISE    = 0.1
	NITER    = 100 
	NPLATEAU = 10
	EPS      = 1.
	RATE     = 0.1
)


func main() {
	flag.Parse()
	switch {
	case flag.NArg() == 0:
	default:
		panic("usage")
	}

	// Data
	// ----
	// Load the data
	var err error
	fmt.Fprint(os.Stderr, "loading...")
	X, Y, err := load()
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, "done")

	if TRAIN == 0 {
		TRAIN = 2./3.
	}
	split := int(math.Round(float64(len(X))*TRAIN))

	// Normalization
	// -------------
	// Use only training data for normalization
	// Normalize X
	start := X[0][0]
	step := (X[split][0] - start) / float64(split-1)
	for i := range X {
		X[i][0] = (X[i][0] - start) / step
	}

	// Normalize Y
	meany, stdy := stat.MeanStdDev(Y[:split], nil)
	for i := range Y {
		Y[i] = (Y[i] - meany) / stdy
	}

	// Problem
	// -------
	// Define the problem
	gpr := &gp.GP{
		NDim:  1,
		Simil: Simil,
		Noise: Noise(NOISE),
	}

	nTheta := gpr.Simil.NTheta() + gpr.Noise.NTheta()

	// Inference
	// ---------
	// Interpolate train and extrapolate test
	fmt.Fprintln(os.Stderr, "Forecasting...")
	fmt.Printf("%s,%s,%s,%s\n", "X", "Y", "mu", "sigma")

	// Construct the initial point in the optimization space
	x := make([]float64, nTheta)
	for i := range x {
		x[i] = 0.1 * rand.NormFloat64()
	}
	gpr.X = X[:split]
	gpr.Y = Y[:split]

	// Optimize the parameters
	opt := &infer.Adam{Rate: RATE}
	iter, ll0, ll := infer.Optimize(
		opt,
		gpr, x,
		NITER, NPLATEAU, EPS)
	fmt.Fprintf(os.Stderr, "%d iterations, %g -> %g\n", iter, ll0, ll)

	mu, sigma, err := gpr.Produce(X)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to forecast: %v\n", err)
	}

	for i := range X {
		fmt.Printf("%g,%g,%g,%g\n",
			(X[i][0]+start)*step,
			meany+stdy*Y[i],
			meany+stdy*mu[i],
			stdy*sigma[i])
	}
	fmt.Fprintln(os.Stderr, "done")
}

// load parses the data from csv and returns inputs and outputs,
// suitable for feeding to the GP.
func load() (
	x [][]float64,
	y []float64,
	err error,
) {
	csv := csv.NewReader(os.Stdin)
RECORDS:
	for {
		record, err := csv.Read()
		switch err {
		case nil:
			// record contains the data
			xi := make([]float64, len(record)-1)
			i := 0
			for ; i != len(record)-1; i++ {
				xi[i], err = strconv.ParseFloat(record[i], 64)
				if err != nil {
					// data error
					return x, y, err
				}
			}
			yi, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				// data error
				return x, y, err
			}
			x = append(x, xi)
			y = append(y, yi)
		case io.EOF:
			// end of file
			break RECORDS
		default:
			// i/o error
			return x, y, err
		}
	}

	return x, y, err
}
