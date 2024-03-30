pipeline {
  agent any

  stages {
    stage('Build Docker Image') {
      steps {
        script {
          dockerImage = docker.build("mlops_assignment_1:${env.BUILD_ID}")
        }
      }
    }
    stage('Push to Docker Hub') {
      steps {
        script {
          docker.withRegistry('', 'docker-hub-credentials') {
            dockerImage.push("latest")
            dockerImage.push("${env.BUILD_ID}")
          }
        }
      }
    }
  }

  post {
    success {
      mail to: 'ahmedfaizan195@gmail.com',
           subject: "SUCCESS: Docker image pushed - Build ${env.BUILD_ID}",
           body: "Jenkins job has successfully pushed the Docker image."
    }
    failure {
      mail to: 'ahmedfaizan195@gmail.com',
           subject: "FAILED: Docker image push - Build ${env.BUILD_ID}",
           body: "Jenkins job has failed to push the Docker image."
    }
  }
}
