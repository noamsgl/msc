all: barebones

GO=go

barebones: kernel/ad/kernel.go main.go 
	$(GO) build -o $@ . 

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./barebones kernel/ad/*.go
