terraform {
  backend "gcs" {
    bucket = "qwiklabs-gcp-04-cacb647e6dab-terraform-state"
    prefix = "dev"
  }
}
