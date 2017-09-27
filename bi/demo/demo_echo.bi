/**
 * Echo a message to terminal.
 * 
 * - `message` : The message.
 */
program demo_echo(message:String <- "") {
  stdout.print(message + "\n");
}
